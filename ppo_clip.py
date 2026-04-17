import time
from random import randrange
from typing import Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
import wandb
import chess_env
from idm_training import Encoder


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    # param count guide:
    #  5x5 : param count 1989 - 2249
    # 10x10: param count 6789 - 7049
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        action_probs: torch.Tensor = torch.softmax(self.fc2(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        value: torch.Tensor = self.fc2(x)
        return value


class ConditionedValueNetwork(nn.Module):
    """Value network for the ASA agent, conditioned on a one-hot environment ID.

    Concatenating the environment ID keeps the total parameter count close to
    that of the per-environment ValueNetwork instances used by the baselines
    (the input layer grows by num_asa_envs, which is small), making comparisons
    of network capacity more meaningful than maintaining a separate ValueNetwork
    per environment, which would multiply parameters by num_asa_envs.
    """
    def __init__(self, state_dim: int, num_envs: int) -> None:
        super(ConditionedValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + num_envs, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, env_id: torch.Tensor) -> torch.Tensor:
        # x:      (batch_size, state_dim)
        # env_id: (batch_size, num_envs) one-hot encoding of active environment
        x = torch.relu(self.fc1(torch.cat([x, env_id], dim=-1)))
        return self.fc2(x)


class ASAPolicy(torch.nn.Module):
    """Action space agnostic (ASA) policy.

    Param count guide:
     5x5 : latent_action_dim=3, nhead=1, dim_feedforward=300 -> 2163 param count
    10x10: latent_action_dim=3, nhead=1, dim_feedforward=975 -> 6888 param count
    """
    def __init__(self, idm: nn.Module, latent_action_dim: int,
                 dim_feedforward: int, state_dim: int,
                 nhead: int = 1, num_layers: int = 1) -> None:
        super().__init__()

        # freeze idm parameters since it shouldn't be trained
        for param in idm.parameters():
            param.requires_grad = False
        self.idm = idm

        assert latent_action_dim % nhead == 0, (
            f"latent_action_dim must be divisible by nhead, but got "
            f"latent_action_dim={latent_action_dim}, nhead={nhead}"
        )

        self.state_dim: int = state_dim

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=latent_action_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.state_embed = torch.nn.Linear(state_dim, latent_action_dim)

        self.Z_A: dict[str, list[torch.Tensor]] = {}
        self.current_env: str = ""

    def make_action_embeddings(self, envs: list[gym.Env] | list[str],
                               device: torch.device) -> None:
        """Build and store latent action embeddings for each environment.

        For each environment and each action index, one transition is sampled
        from a random starting state and passed through the frozen IDM encoder.
        The resulting latent vector is stored as the embedding for that action.

        VERY IMPORTANT: this is probably a massive area for improvement. Only a
        single transition is used to make the latent actions — a single
        random-start transition may not be representative of what the action
        does across all starting positions.

        Args:
            envs:   List of gym.Env instances or env name strings to embed.
            device: Device to run the IDM encoder on.
        """
        # handle envs as name strings case
        if type(envs[0]) == str:  # assumes all elements are either gym.Envs or strings
            gym_envs: list[gym.Env] = [gym.make(env_name) for env_name in envs]
        else:
            gym_envs = envs  # type: ignore[assignment]

        self.Z_A = {
            env.unwrapped.spec.id.replace('chess_env/', ''): []
            for env in gym_envs
        }
        self.current_env = gym_envs[0].unwrapped.spec.id.replace('chess_env/', '')

        for env in gym_envs:
            action_embeddings: list[torch.Tensor] = []

            # use idm to make action embeddings. Add them in order of action indices.
            for i in range(env.action_space.n):
                s_t: np.ndarray = env.reset()[0]
                s_tp1: np.ndarray = env.step(i)[0]
                transition: torch.Tensor = torch.stack(
                    [torch.tensor(s_t), torch.tensor(s_tp1)]
                ).to(torch.float32).to(device)  # shape (2, height, width)
                action_embeddings.append(self.idm(transition))

            self.Z_A[env.unwrapped.spec.id.replace('chess_env/', '')] = action_embeddings

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute a probability distribution over actions for each state in s.

        Args:
            s: Tensor of shape (batch_size, state_size) representing a batch of states.

        Returns:
            Tensor of shape (batch_size, num_actions) — probability distribution.
        """
        Z_A: list[torch.Tensor] = self.Z_A[self.current_env]

        s = s.clone().detach().to(torch.float32)
        if s.dim() == 1:
            # add explicit batch dimension
            s = s.unsqueeze(0)

        # flatten everything after batch dim
        s = s.flatten(1)

        assert Z_A != [], "expected Z_A to be a list of latent actions, but got empty list"
        assert s.dim() == 2 and s.shape[1] == self.state_dim, (
            f"expected s to be a batch of states with shape (batch_size, state_size), "
            f"but got shape {s.shape}"
        )

        # stack and make contiguous to avoid creating non-contiguous views later
        Z_A_tensor: torch.Tensor = torch.cat(Z_A).contiguous()  # (num_actions, latent_action_dim)

        embedded_s: torch.Tensor = self.state_embed(s)  # (batch_size, latent_action_dim)

        batched_embedded_s: torch.Tensor = embedded_s.view(
            embedded_s.shape[0], 1, embedded_s.shape[1]
        )  # (batch_size, 1, latent_action_dim)

        # FIXME: "Z_A_tensor.repeat(batched_embedded_s.shape[0], 1)" caused
        # "RuntimeError: Number of dimensions of repeat dims can not be smaller
        # than number of dimensions of tensor". Fixed by adding an additional 1.
        Z_A_to_cat: torch.Tensor = (
            Z_A_tensor
            .repeat(batched_embedded_s.shape[0], 1)
            .view(batched_embedded_s.shape[0], Z_A_tensor.shape[0], Z_A_tensor.shape[1])
            .contiguous()
        )  # (batch_size, num_actions, latent_action_dim)

        s_and_Z_A: torch.Tensor = torch.cat(
            [batched_embedded_s, Z_A_to_cat], dim=1
        ).contiguous()  # (batch_size, num_actions + 1, latent_action_dim)

        contextual_embeddings: torch.Tensor = self.transformer_encoder(s_and_Z_A)
        # (batch_size, num_actions + 1, latent_action_dim)

        z_i: torch.Tensor = contextual_embeddings.mean(dim=1)  # (batch_size, latent_action_dim)

        probability_distribution: torch.Tensor = self.decoder(z_i, Z_A_tensor)
        # (batch_size, num_actions)

        return probability_distribution

    def decoder(self, z_i: torch.Tensor, Z_A: torch.Tensor) -> torch.Tensor:
        """Compute action probabilities via cosine similarity.

        Given a context vector z_i, compute cosine similarity between z_i and
        each action embedding in Z_A, then convert to a probability distribution.

        Cosine similarity is the chosen (non-learned) similarity function.
        Alternatives worth considering: dot product (unnormalized, magnitude-
        sensitive), negative Euclidean distance, RBF/Gaussian kernel.

        Args:
            z_i: Tensor of shape (batch_size, latent_action_dim).
            Z_A: Tensor of shape (num_actions, latent_action_dim).

        Returns:
            Probability distribution of shape (batch_size, num_actions).
        """
        # make repeated tensors contiguous before view/reshape to avoid view aliasing issues
        a: torch.Tensor = z_i.repeat_interleave(Z_A.shape[0], dim=0).contiguous()
        b: torch.Tensor = Z_A.repeat(z_i.shape[0], 1).contiguous()
        # calculate cosine similarities in batch
        logits: torch.Tensor = torch.cosine_similarity(a, b).view(z_i.shape[0], -1)

        # logits + 1 shifts cosine similarities from [-1, 1] to [0, 2] to avoid negative values
        probability_distribution: torch.Tensor = (
            (logits + 1) / torch.sum(logits + 1, dim=1, keepdim=True)
        )  # (batch_size, num_actions)

        return probability_distribution

    def evaluate(self, x: torch.Tensor) -> int:
        return int(self.forward(x).argmax())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def ppo_clip(policy_net: nn.Module, value_net: nn.Module,
             policy_optimizer: optim.Optimizer,
             value_optimizer: optim.Optimizer,
             envs: list[gym.Env],
             config: dict,
             device: torch.device,
             agent_name: str = "agent",
             asa_env_names: Optional[list[str]] = None) -> int:
    """Run PPO-Clip training for one agent.

    All metrics are logged to W&B under the namespace
    ``{agent_name}/{env_name}/{metric}``.

    Args:
        policy_net:        The policy network to train.
        value_net:         The value network to train. Pass a
                           ConditionedValueNetwork when asa_env_names is set.
        policy_optimizer:  Optimizer for policy_net.
        value_optimizer:   Optimizer for value_net.
        envs:              List of environments to train in (sampled randomly
                           each episode).
        config:            Full experiment config dict.
        device:            Device to run training on.
        agent_name:        Namespace prefix for W&B metric keys.
        asa_env_names:     If provided, policy_net is treated as an ASAPolicy
                           and value_net must be a ConditionedValueNetwork.
                           Used to build per-step one-hot env ID vectors.

    Returns:
        Total number of training epochs completed.
    """
    print(f"starting ppo_clip for {[env.unwrapped.spec.id.replace('chess_env/', '') for env in envs]}")

    total_timesteps: int  = config["total_timesteps"]
    gamma: float          = config["gamma"]
    clip_epsilon: float   = config["clip_epsilon"]
    eval_frequency: int   = config["eval_frequency"]
    eval_episodes: int    = config["eval_episodes"]

    # Whether this is an ASA agent with a ConditionedValueNetwork
    is_asa: bool = asa_env_names is not None

    timesteps_taken: int = 0
    epoch: int = 0

    while timesteps_taken < total_timesteps:
        env: gym.Env = envs[randrange(len(envs))]

        # Set the current environment on the policy if it's an ASA agent
        try:
            policy_net.current_env = env.unwrapped.spec.id.replace('chess_env/', '')
        except AttributeError:
            pass  # baseline PolicyNetwork has no current_env attribute

        states: list[np.ndarray]  = []
        actions: list[int]        = []
        rewards: list[float]      = []
        log_probs_old: list[float] = []
        env_ids: list[torch.Tensor] = []  # only populated for ASA agents

        # Build env one-hot for the current episode (same for all steps)
        if is_asa:
            env_idx: int = asa_env_names.index(  # type: ignore[union-attr]
                env.unwrapped.spec.id.replace('chess_env/', '')
            )
            env_id_onehot: torch.Tensor = torch.zeros(
                1, len(asa_env_names), device=device  # type: ignore[arg-type]
            )
            env_id_onehot[0, env_idx] = 1.0

        state: np.ndarray = env.reset(seed=config["seed"] + epoch)[0].flatten()
        done: bool = False

        while not done:
            state_tensor: torch.Tensor = torch.FloatTensor(state).unsqueeze(0).to(device=device)
            action_probs: torch.Tensor = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action: torch.Tensor = action_dist.sample()
            log_prob: torch.Tensor = action_dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = next_state.flatten()
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs_old.append(log_prob.item())
            if is_asa:
                env_ids.append(env_id_onehot)

            state = next_state

        # Calculate discounted returns
        returns: list[float] = []
        discounted_return: float = 0.0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)
        returns_tensor: torch.Tensor = torch.FloatTensor(returns).to(device=device)

        # convert to numpy arrays first to silence the warning about slowness
        # of converting a list of np.ndarrays to torch.Tensor
        states_tensor: torch.Tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor: torch.Tensor = torch.LongTensor(np.array(actions)).to(device)
        log_probs_old_tensor: torch.Tensor = torch.FloatTensor(np.array(log_probs_old)).to(device)

        # Calculate advantages
        if is_asa:
            env_ids_tensor: torch.Tensor = torch.cat(env_ids, dim=0)  # (T, num_asa_envs)
            values: torch.Tensor = value_net(states_tensor, env_ids_tensor).squeeze(-1)
        else:
            values = value_net(states_tensor).squeeze(-1)

        advantages: torch.Tensor = returns_tensor - values.detach()

        # Update policy
        action_probs = policy_net(states_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs: torch.Tensor = action_dist.log_prob(actions_tensor)
        ratios: torch.Tensor = torch.exp(log_probs - log_probs_old_tensor)
        surr1: torch.Tensor = ratios * advantages
        surr2: torch.Tensor = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update value function
        value_loss: torch.Tensor = nn.MSELoss()(values, returns_tensor)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        timesteps_taken += len(rewards)
        epoch += 1

        # eval once every eval_frequency epochs
        if epoch % eval_frequency == 0:
            for environment in envs:
                ep_return, ep_len = evaluate(policy_net, environment, eval_episodes, device)
                env_tag: str = environment.unwrapped.spec.id.replace('chess_env/', '')
                wandb.log({
                    f"{agent_name}/{env_tag}/return":      ep_return,
                    f"{agent_name}/{env_tag}/ep_len":      ep_len,
                    f"{agent_name}/{env_tag}/policy_loss": policy_loss.item(),
                    f"{agent_name}/{env_tag}/value_loss":  value_loss.item(),
                    f"{agent_name}/timestep":              timesteps_taken,
                    f"{agent_name}/epoch":                 epoch,
                })

    # eval one last time after training is done
    for environment in envs:
        ep_return, ep_len = evaluate(policy_net, environment, eval_episodes, device)
        env_tag = environment.unwrapped.spec.id.replace('chess_env/', '')
        wandb.log({
            f"{agent_name}/{env_tag}/return":      ep_return,
            f"{agent_name}/{env_tag}/ep_len":      ep_len,
            f"{agent_name}/{env_tag}/policy_loss": policy_loss.item(),
            f"{agent_name}/{env_tag}/value_loss":  value_loss.item(),
            f"{agent_name}/timestep":              timesteps_taken,
            f"{agent_name}/epoch":                 epoch,
        })

    return epoch


def evaluate(policy: nn.Module, env: gym.Env, n_episodes: int,
             device: torch.device) -> tuple[float, float]:
    """Evaluate a policy by running n_episodes episodes stochastically.

    Args:
        policy:     Policy network to evaluate.
        env:        Environment to evaluate in.
        n_episodes: Number of episodes to average over.
        device:     Device the policy is on.

    Returns:
        Tuple of (avg_return, avg_episode_length).
    """
    try:
        policy.current_env = env.unwrapped.spec.id.replace('chess_env/', '')
    except AttributeError:
        pass  # baseline PolicyNetwork has no current_env attribute

    total_rewards: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(n_episodes):
        rewards: list[float] = []
        state: np.ndarray = env.reset()[0].flatten()
        done: bool = False

        while not done:
            state_tensor: torch.Tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs: torch.Tensor = policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action: torch.Tensor = action_dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = next_state.flatten()
            done = terminated or truncated

            rewards.append(reward)
            state = next_state

        total_rewards.append(sum(rewards))
        episode_lengths.append(len(rewards))

    avg_return: float   = sum(total_rewards) / len(total_rewards)
    avg_ep_len: float   = sum(episode_lengths) / len(episode_lengths)
    return avg_return, avg_ep_len


# ---------------------------------------------------------------------------
# Multi-process training workers
# ---------------------------------------------------------------------------

def _train_worker(gpu_id: int, rank_id: int, config: dict, agent_spec: dict,
                  run_id: str, run_dir: str) -> None:
    """Worker function executed in a child process for one agent.

    Each worker initializes its own wandb context joined to the parent run so
    that all metrics land in the same W&B run page.

    Args:
        gpu_id:     GPU index to run on. Multiple agents may share one GPU;
                    assignment is round-robin in the caller.
        config:     Full experiment config dict.
        agent_spec: Dict describing the agent to train:
                      {"type": "baseline"|"asa"|"asa_transfer",
                       "env_names": [str, ...],
                       "agent_name": str}
        run_id:     Parent W&B run ID. Workers join this run so metrics are
                    aggregated on one run page.
        run_dir:    Local directory for saving model checkpoints.
    """

    # time.sleep(rank_id * 120) # stagger wandb.inits between instances of _train_worker

    device: torch.device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(config["seed"])

    # Join the parent W&B run so all agents log to the same run
    wandb.init(
        project=config.get("project_name"),
        entity=config.get("wandb_entity"),
        group=run_id,
        job_type=agent_spec["agent_name"],
        config=config,
        resume="allow",
        # settings=wandb.Settings(init_timeout=120)
    )

    envs: list[gym.Env] = [
        gym.make(name, width=config["width"], height=config["height"])
        for name in agent_spec["env_names"]
    ]
    state_dim: int = config["height"] * config["width"]
    agent_name: str = agent_spec["agent_name"]

    if agent_spec["type"] == "baseline":
        env: gym.Env = envs[0]
        policy: nn.Module = PolicyNetwork(state_dim, env.action_space.n).to(device)
        value_net: nn.Module = ValueNetwork(state_dim).to(device)
        policy_opt = optim.Adam(policy.parameters(),    lr=config["ppo_lr"])
        value_opt  = optim.Adam(value_net.parameters(), lr=config["ppo_lr"])
        ppo_clip(policy, value_net, policy_opt, value_opt,
                 envs, config, device, agent_name=agent_name)

    elif agent_spec["type"] == "asa":
        idm: Encoder = Encoder(
            height=config["height"],
            width=config["width"],
            input_dim=state_dim,
            latent_dim=config["idm_latent_dim"]
        ).to(device)
        idm.load_state_dict(torch.load(
            Path(run_dir) / "idm.pt", map_location=device, weights_only=False
        ))
        for p in idm.parameters():
            p.requires_grad = False

        asa_env_names: list[str] = [
            e.replace('chess_env/', '') for e in agent_spec["env_names"]
        ]
        policy = ASAPolicy(
            idm,
            latent_action_dim=config["latent_action_dim"],
            dim_feedforward=config["dim_feedforward"],
            state_dim=state_dim,
            nhead=config["transformer_nhead"],
            num_layers=config["transformer_layers"],
        ).to(device)
        policy.make_action_embeddings(envs, device)

        value_net = ConditionedValueNetwork(state_dim, len(envs)).to(device)
        policy_opt = optim.Adam(policy.parameters(),    lr=config["ppo_lr"])
        value_opt  = optim.Adam(value_net.parameters(), lr=config["ppo_lr"])
        ppo_clip(policy, value_net, policy_opt, value_opt,
                 envs, config, device,
                 agent_name=agent_name,
                 asa_env_names=asa_env_names)

    elif agent_spec["type"] == "asa_transfer":
        idm = Encoder(
            height=config["height"],
            width=config["width"],
            input_dim=state_dim,
            latent_dim=config["idm_latent_dim"]
        ).to(device)
        idm.load_state_dict(torch.load(
            Path(run_dir) / "idm.pt", map_location=device, weights_only=False
        ))
        for p in idm.parameters():
            p.requires_grad = False

        asa_env_names = [
            e.replace('chess_env/', '') for e in agent_spec["env_names"]
        ]
        # Build policy and load the trained ASA checkpoint.
        # Each transfer worker loads the same checkpoint independently —
        # they do not share or modify each other's state.
        policy = ASAPolicy(
            idm,
            latent_action_dim=config["latent_action_dim"],
            dim_feedforward=config["dim_feedforward"],
            state_dim=state_dim,
            nhead=config["transformer_nhead"],
            num_layers=config["transformer_layers"],
        ).to(device)
        policy.load_state_dict(torch.load(
            Path(run_dir) / "asa_agent_policy.pt",
            map_location=device,
            weights_only=False
        ))
        policy.make_action_embeddings(envs, device)

        value_net = ConditionedValueNetwork(state_dim, len(envs)).to(device)
        policy_opt = optim.Adam(policy.parameters(),    lr=config["ppo_lr"])
        value_opt  = optim.Adam(value_net.parameters(), lr=config["ppo_lr"])
        ppo_clip(policy, value_net, policy_opt, value_opt,
                 envs, config, device,
                 agent_name=agent_name,
                 asa_env_names=asa_env_names)

    elif agent_spec["type"] == "baseline_transfer":
        # Load a trained baseline policy and fine-tune it on an unseen environment
        # whose action space size matches the source baseline's action space.
        # agent_spec["source_env_name"] identifies which checkpoint to load.
        source_tag: str = (
            agent_spec["source_env_name"]
            .replace("chess_env/", "")
            .replace("-v0", "")
        )
        policy = PolicyNetwork(state_dim, envs[0].action_space.n).to(device)
        value_net: nn.Module = ValueNetwork(state_dim).to(device)
        policy.load_state_dict(torch.load(
            Path(run_dir) / f"{source_tag}_baseline_policy.pt",
            map_location=device,
            weights_only=False
        ))
        value_net.load_state_dict(torch.load(
            Path(run_dir) / f"{source_tag}_baseline_value.pt",
            map_location=device,
            weights_only=False
        ))
        policy_opt = optim.Adam(policy.parameters(),    lr=config["ppo_lr"])
        value_opt  = optim.Adam(value_net.parameters(), lr=config["ppo_lr"])
        ppo_clip(policy, value_net, policy_opt, value_opt,
                 envs, config, device, agent_name=agent_name)

    else:
        raise ValueError(f"Unknown agent type: {agent_spec['type']!r}")

    # Save policy and value network checkpoints
    save_name: str = agent_spec["agent_name"]
    torch.save(policy.state_dict(),    Path(run_dir) / f"{save_name}_policy.pt")
    torch.save(value_net.state_dict(), Path(run_dir) / f"{save_name}_value.pt")

    wandb.finish()


def run_all_agents(config: dict, run_dir: str) -> None:
    """Spawn one process per agent, assigned to GPUs round-robin.

    Trains one baseline agent per environment in training_envs, plus one ASA
    agent that trains across asa_envs. Workers are assigned GPUs round-robin
    (gpu_id = rank % num_gpus), so this works correctly with any number of
    GPUs — including fewer GPUs than agents.

    Args:
        config:  Full experiment config dict.
        run_dir: Directory containing idm.pt and where checkpoints are saved.
    """
    wandb.setup() # used because _train_worker initiates a run in a spawned instance (see https://docs.wandb.ai/models/track/log/distributed-training)

    num_gpus: int = torch.cuda.device_count() or 1  # fall back to CPU if no GPUs

    all_env_names: list[str] = config["all_envs"]
    asa_env_names: list[str] = config["asa_envs"]

    # One baseline agent per environment
    baseline_specs: list[dict] = [
        {
            "type":       "baseline",
            "env_names":  [name],
            "agent_name": name.replace("chess_env/", "").replace("-v0", "") + "_baseline",
        }
        for name in all_env_names
    ]

    # One ASA agent training on asa_envs
    asa_spec: dict = {
        "type":       "asa",
        "env_names":  asa_env_names,
        "agent_name": "asa_agent",
    }

    agent_specs: list[dict] = baseline_specs + [asa_spec]
    run_id: str = wandb.run.id if wandb.run is not None else "local"
    processes: list[mp.Process] = []

    for rank, spec in enumerate(agent_specs):
        gpu_id: int = rank % num_gpus
        p = mp.Process(
            target=_train_worker,
            args=(gpu_id, rank, config, spec, run_id, run_dir)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _action_space_size(env_name: str, cfg: dict) -> int:
    """Return the action space size for env_name without keeping the env open.

    Args:
        env_name: Full gym env name (e.g. "chess_env/KingWorld-v0").
        cfg:      Experiment config dict (must contain "width" and "height").

    Returns:
        Integer number of discrete actions.
    """
    env: gym.Env = gym.make(env_name, width=cfg["width"], height=cfg["height"])
    n: int = env.action_space.n
    env.close()
    return n


def run_asa_transfer(config: dict, run_dir: str) -> None:
    """For each unseen environment, independently load the trained ASA policy
    checkpoint and train on that single environment, recording the learning curve.

    This project is not concerned with continual learning, so each unseen
    environment is evaluated independently — the asa_agent_policy.pt checkpoint
    is loaded fresh for each worker, and workers do not share state. This gives
    a clean jumpstart / learning-speed measurement for each environment without
    any cross-contamination between transfer tasks.

    The unseen environments include at least one with an action space size not
    present in asa_envs (e.g. GoldGeneralWorld with 7 actions), testing true
    action-space generalization.

    Args:
        config:  Full experiment config dict.
        run_dir: Directory containing asa_agent_policy.pt and idm.pt.
    """
    wandb.setup() # used because _train_worker initiates a run in a spawned instance (see https://docs.wandb.ai/models/track/log/distributed-training)
    
    num_gpus: int = torch.cuda.device_count() or 1

    unseen_envs: list[str] = [
        name for name in config["all_envs"]
        if name not in config["asa_envs"]
    ]

    transfer_specs: list[dict] = [
        {
            "type":       "asa_transfer",
            "env_names":  [name],
            "agent_name": f"asa_transfer_{name.replace('chess_env/', '').replace('-v0', '')}",
        }
        for name in unseen_envs
    ]

    run_id: str = wandb.run.id if wandb.run is not None else "local"
    processes: list[mp.Process] = []

    for rank, spec in enumerate(transfer_specs):
        gpu_id: int = rank % num_gpus
        p = mp.Process(
            target=_train_worker,
            args=(gpu_id, rank, config, spec, run_id, str(run_dir))
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def run_baseline_transfer(config: dict, run_dir: str) -> None:
    """For each unseen environment, find all trained baseline agents whose action
    space size matches and fine-tune each one on that environment independently.

    Matching is done at runtime by comparing ``action_space.n`` between each
    training env (in ``config["asa_envs"]``) and each unseen env (in
    ``config["all_envs"]`` but not ``config["asa_envs"]``). This keeps the logic
    correct across grid-size sweeps or env additions without hardcoding.

    If no trained baseline has a matching action space for a given unseen env,
    a warning is printed and that env is skipped — no error is raised.

    Each (source_baseline, target_env) pair produces one independent worker that
    runs the full ``total_timesteps`` PPO budget starting from the trained
    baseline checkpoint. Workers do not share state.

    Metrics are logged to W&B under the namespace:
        ``baseline_transfer_{source_tag}_to_{target_tag}/{target_tag}/{metric}``

    Args:
        config:  Full experiment config dict.
        run_dir: Directory containing ``{source_tag}_baseline_policy.pt`` and
                 ``{source_tag}_baseline_value.pt`` checkpoints from stage 2.
    """
    wandb.setup()  # required for W&B in spawned child processes

    num_gpus: int = torch.cuda.device_count() or 1

    # Map action space size -> list of training env names with that size.
    size_to_source: dict[int, list[str]] = {}
    for name in config["asa_envs"]:
        n: int = _action_space_size(name, config)
        size_to_source.setdefault(n, []).append(name)

    unseen_envs: list[str] = [
        name for name in config["all_envs"]
        if name not in config["asa_envs"]
    ]

    transfer_specs: list[dict] = []
    for target_name in unseen_envs:
        target_n: int = _action_space_size(target_name, config)
        sources: list[str] = size_to_source.get(target_n, [])
        if not sources:
            print(
                f"[baseline_transfer] No matching baseline for {target_name} "
                f"(action_space.n={target_n}). Skipping."
            )
            continue
        for source_name in sources:
            source_tag: str = source_name.replace("chess_env/", "").replace("-v0", "")
            target_tag: str = target_name.replace("chess_env/", "").replace("-v0", "")
            transfer_specs.append({
                "type":            "baseline_transfer",
                "env_names":       [target_name],
                "source_env_name": source_name,
                "agent_name":      f"baseline_transfer_{source_tag}_to_{target_tag}",
            })

    if not transfer_specs:
        print("[baseline_transfer] No transfer pairs found. Nothing to do.")
        return

    run_id: str = wandb.run.id if wandb.run is not None else "local"
    processes: list[mp.Process] = []

    for rank, spec in enumerate(transfer_specs):
        gpu_id: int = rank % num_gpus
        p = mp.Process(
            target=_train_worker,
            args=(gpu_id, rank, config, spec, run_id, str(run_dir))
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

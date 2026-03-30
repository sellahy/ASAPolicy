# train policy on 2 of remaining chessworld envs by PPO:
# chess_env/KingWorld-v0, chess_env/KnightWorld-v0, 
# chess_env/SilverGeneralWorld-v0, chess_env/ZebraWorld-v0

# compare to simpler parametric policies trained per environment. compare...
# asymptotic performance (how much better return does it see than a policy without transfer? compare test results throughout training and look at difference after convergence)
# learning speed improvement (how much faster does it converge than a policy without transfer? ^ will show this as well)
# jumpstart (how much better does it initially do than a randomly initialized policy without transfer? train in 1 of the remaining 2 environments to measure this)
# 0-shot transfer (measure performance in remaining environment. Compare to random policy for reference, but hope for similar performance to other envs)

# TODO: move any remaining hyperparams to config

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import namedtuple
from idm_training import Encoder
from config import policy_training_config_dict as config
import wandb
import pdb
from math import floor

Agent = namedtuple('Agent', ["policy_network", "value_network", "policy_optimizer", "value_optimizer"])

def clean_env_names(names_to_clean : list[str]) -> str:
    env_names : list[str] = sorted([env for env in names_to_clean]) # alphabetized list of env names
    cleaned_envs_name : list[str] = sorted([env_name[str.rfind(env_name, '/')+1:] for env_name in env_names]) # alphabetized list of env names containing only the portion after any /'s
    name_prefix : str = "and".join(cleaned_envs_name) # env names joined by "and". To be used with logging
    return name_prefix

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=config["height"]*config["width"], output_dim=9):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, output_dim)

    def forward(self, x):
        # assume x is a tensor with shape (n, 10, 10)
        x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 2:
            x = torch.flatten(x, start_dim=0)
        else:
            # preserve batch dim if present
            x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs
    
    def evaluate(self, x) -> int:
        return int(self.forward(x).argmax())


# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim=config["height"]*config["width"]):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 2:
            x = torch.flatten(x, start_dim=0)
        else:
            # preserve batch dim if present
            x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class ASAgnosticPolicy(torch.nn.Module):
    def __init__(self, env_names : list[str], latent_action_dim, state_dim, nhead=3, num_layers=1):
        super().__init__()
        
        assert latent_action_dim % nhead == 0, f"latent_action_dim must be divisible by nhead, but got latent_action_dim={latent_action_dim}, nhead={nhead}"

        self.state_dim = state_dim

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=latent_action_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers)
    
        self.state_embed = torch.nn.Linear(state_dim, latent_action_dim)

        self.Z_A : dict[str, list[torch.Tensor]] = {env_name : [] for env_name in env_names}
        self.current_env : str = env_names[0]

        # TODO: probably should move Z_A set up to this function. Would need to include idm as argument and save it as a member var

    def forward(self, s) -> torch.Tensor:
        """Given a list of latent actions Z_A and a batch of states s, returns the transformer encoder output.
        Z_A: list of torch.Tensors representing latent actions corresponding to all actions in self.A
        s: tensor of shape (batch_size, state_size) representing a batch of states.
        Returns: torch.Tensor of shape (batch_size, latent_action_dim) representing the log probabilities of each action for each state in batch."""
        # assume Z_A is a list of latent actions (torch.Tensors) corresponding to all actions in self.A
        # assume s is a batch of states (torch.Tensor) of shape (batch_size, state_size)
        Z_A = self.Z_A[self.current_env]

        # expect input to be (batch_size, height, width). need to reshape input into correct shape
    
        s = torch.tensor(s, dtype=torch.float32)
        if s.dim() == 2:
            # add explicit batch dimension
            s = s.unsqueeze(0)
        
        # flatten everything after batch dim
        s = s.flatten(1)

        assert Z_A != [], "expected Z_A to be a list of latent actions, but got empty list"
        assert s.dim() == 2 and s.shape[1] == self.state_dim, f"expected s to be a batch of states with shape (batch_size, state_size), but got shape {s.shape}"
        # stack and make contiguous to avoid creating non-contiguous views later
        # Z_A_tensor = torch.stack(Z_A).contiguous() # shape (num_actions, latent_action_dim)
        Z_A_tensor = torch.cat(Z_A).contiguous() # shape (num_actions, latent_action_dim)

        embedded_s = self.state_embed(s) # shape (batch_size, latent_action_dim)

        batched_embedded_s = embedded_s.view(embedded_s.shape[0], 1, embedded_s.shape[1]) # shape (batch_size, 1, latent_action_dim)

        # breakpoint()

        # FIXME: "Z_A_tensor.repeat(batched_embedded_s.shape[0], 1)" caused "RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor", I fixed this by adding an additional 1, but I don't know if this preserves the intended purpose of this line
        # repeat then reshape, ensure contiguous to avoid aliasing issues inside transformer
        Z_A_to_cat = Z_A_tensor.repeat(batched_embedded_s.shape[0], 1).view(batched_embedded_s.shape[0], Z_A_tensor.shape[0], Z_A_tensor.shape[1]).contiguous() # shape (batch_size, num_actions, latent_action_dim)

        s_and_Z_A = torch.cat([batched_embedded_s, Z_A_to_cat], dim=1).contiguous() # shape (batch_size, num_actions + 1, latent_action_dim)

        contextual_embeddings = self.transformer_encoder(s_and_Z_A)  # shape (batch_size, num_actions + 1, latent_action_dim)

        z_i = contextual_embeddings.mean(dim = 1)  # shape (batch_size, latent_action_dim)

        probability_distribution = self.decoder(z_i, Z_A_tensor)  # shape (batch_size, num_actions)

        return probability_distribution
    
    def decoder(self, z_i: torch.Tensor, Z_A : torch.Tensor) -> torch.Tensor:
        # Given an input latent state z_i, compute probabilities by computing 
        # cosine similarity between z_i and all z_a in Z_A and converting to 
        # probability distribution.
        # returns probability distribution over all actions (ex. prob_dist[0] 
        # corresponds to probability for action A[0])
        # assume z_i has shape (batch_size, latent_action_dim)
    
        # make repeated tensors contiguous before view/reshape to avoid view aliasing issues
        a = z_i.repeat_interleave(Z_A.shape[0], dim=0).contiguous()
        b = Z_A.repeat(z_i.shape[0], 1).contiguous()
        # calculate cosine similarities in batch
        logits = torch.cosine_similarity(a, b).view(z_i.shape[0], -1)
        
        # logits + 1 shifts cosine similarities from [-1, 1] to [0, 2] to avoid negative values
        probability_distribution = (logits + 1)/torch.sum(logits + 1, dim=1, keepdim=True)  # shape (batch_size, num_actions)
        
        return probability_distribution # shape (batch_size, num_actions)

    def evaluate(self, x) -> int:
        return int(self.forward(x).argmax())

# https://www.codegenes.net/blog/ppo-pytorch/
def ppo_clip(policy_net : nn.Module, value_net : nn.Module, 
             policy_optimizer, value_optimizer, 
             envs : list[gym.Env], epochs : int, run) -> nn.Module:
    policy_net.train()
    value_net.train()

    name_prefix : str = clean_env_names([env.spec.id for env in envs])
    
    for epoch in range(epochs):
        env = envs[random.randint(0, len(envs)-1)]
        try:
            # set current env for action space agnostic policy
            policy_net.current_env = env.spec.id
        except AttributeError:
            print("continuing with simple policy")
        states, actions, rewards, log_probs_old = [], [], [], []
        state = env.reset()
        state = state[0]['matrix']
        terminated = False
        truncated = False
        while not (terminated or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device=device)
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            try:
                next_state_dict, reward, terminated, truncated, _ = env.step(action.item())
            except KeyError:
                breakpoint()
            next_state : np.ndarray = next_state_dict["matrix"]

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs_old.append(log_prob.item())

            state = next_state

        # Calculate returns
        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.FloatTensor(returns).to(device=device)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(device)

        # Calculate advantages
        values = value_net(states).squeeze().to(device)
        advantages = returns - values.detach().to(device)

        # Update policy
        action_probs = policy_net(states).to(device)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions).to(device)
        ratios = torch.exp(log_probs - log_probs_old).to(device)
        surr1 = ratios * advantages.to(device)
        surr2 = (torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages).to(device)
        policy_loss = -torch.min(surr1, surr2).mean().to(device)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update value function
        value_loss = nn.MSELoss()(values, returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # model saving
        if epoch % 5 == 0:
            torch.save(policy_net.state_dict(), f"models/{name_prefix}_policy.pt")
        
        evaluate(policy_net=policy_net, policy_name=f"{name_prefix}_policy", envs=envs, run=run, epoch=epoch)

        # log loss
        policy_name : str = f"{name_prefix}_policy"
        value_network_name : str = f"{name_prefix}_value_net"
        
        # add data for logging
        if policy_name not in losses_data.keys():
            losses_data[policy_name] = []
        losses_data[policy_name].append([epoch, policy_loss.item()])

        if value_network_name not in losses_data.keys():
            losses_data[value_network_name] = []
        losses_data[value_network_name].append([epoch, value_loss.item()])
        # run.log({"policy name":        f"{name_prefix}_policy",    "policy loss" :        policy_loss.item(), "epoch" : epoch})
        # run.log({"value network name": f"{name_prefix}_value_net", "value network loss" : value_loss.item(),  "epoch" : epoch})

    return policy_net

def evaluate(policy_net : nn.Module, policy_name : str, envs : list[gym.Env], run, epoch : int):
    policy_net.eval() # set policy_net to eval mode for performance logging, then return to training mode
    rollouts_to_avg_over = 10
    for env in envs:
        
        # set policy's current env
        env_name : str = env.spec.id
        try:
            # set current env for action space agnostic policy
            policy_net.current_env = env_name
        except AttributeError:
            print("continuing with simple policy")
        
        cumulative_rewards = [] # for storing cumulative rewards per episode
        timesteps = [] # for storing num of timesteps per episode
        for i in range(rollouts_to_avg_over):
            o = env.reset()[0]["matrix"]
            o = torch.tensor(o).to(device=device)
            cumulative_reward = 0
            timestep = 0
            terminated = False
            truncated = False

            # rollout trajectory
            while not (terminated or truncated):
                o, reward, terminated, truncated, _ = env.step(policy_net.evaluate(o))
                o = o["matrix"]
                o = torch.tensor(o).to(device=device)
                cumulative_reward += reward
                timestep += 1
            cumulative_rewards.append(cumulative_reward)
            timesteps.append(timestep)
        avg_cumulative_reward = np.average(cumulative_rewards)
        std_dev_rewards = np.std(cumulative_rewards)
        avg_num_timesteps = np.average(timesteps)
        std_dev_num_timesteps = np.std(timesteps)
        
        # add data for logging
        if policy_name not in rewards_data.keys():
            rewards_data[policy_name] = []
        rewards_data[policy_name].append([epoch, avg_cumulative_reward])

        if policy_name not in reward_std_devs_data.keys():
            reward_std_devs_data[policy_name] = []
        reward_std_devs_data[policy_name].append([epoch, std_dev_rewards])

        if policy_name not in timesteps_data.keys():
            timesteps_data[policy_name] = []
        timesteps_data[policy_name].append([epoch, avg_num_timesteps])

        if policy_name not in timestep_std_devs_data.keys():
            timestep_std_devs_data[policy_name] = []
        timestep_std_devs_data[policy_name].append([epoch, std_dev_num_timesteps])
        
        # run.log({"policy_name" : policy_name, "env" : env_name, "average cumulative reward" :   avg_cumulative_reward, "reward standard deviation" :   std_dev_rewards,         "epoch" : epoch})
        # run.log({"policy_name" : policy_name, "env" : env_name, "average number of timesteps" : avg_num_timesteps,     "timestep standard deviation" : std_dev_num_timesteps,   "epoch" : epoch})
    
    # return policy_net to training mode
    policy_net.train()


# helper functions
def moving_avg(data : list, window : int):
    # computes centered moving average of data and returns array of moving averages
    # requires window to be odd
    weights = np.ones(window)/window
    return np.convolve(data, weights, mode='valid')

def get_moving_avg(data : dict[str, list], window : int):
    # {network_name : [[epoch, loss], ...], ...} -> {network_name : [[epoch, moving_avg_loss], ...], ...}
    moving_avg_data : dict[str, list] = {}
    for network_name in data.keys():
        values = data[network_name].copy() # this is a list of tuples of epochs and values (2 element lists)
        data_to_avg = [epoch_loss_tuple[1] for epoch_loss_tuple in values] # unnested list of just the values
        window = min(window, len(data_to_avg)) # ensure window is not larger than size of data array
        avged_losses = moving_avg(data_to_avg, window) # compute moving avg
        moving_avg_corresponding_epochs = [epoch_loss_tuple[0] for epoch_loss_tuple in values] # unnested list of just the epochs
        corresponding_epochs = moving_avg_corresponding_epochs[floor(window/2):len(moving_avg_corresponding_epochs)-floor(window/2)] # shave off a half window's worth of epochs on either side
        moving_avg_data[network_name] = [[corresponding_epochs[i], avged_losses[i]] for i in range(len(corresponding_epochs))]
    return moving_avg_data

def wandb_log_data(data_dict : dict[str, list], columns : list[str], run, title, chart_id):
    # columns[0] is x label, columns[1] is y label
    # title should include "network_name" for replacement with name of network
    for network_name, data in losses_data.items():
        table = wandb.Table(data=data, columns=columns)
        network_specific_title = title.replace("network_name", network_name)
        print(network_specific_title)
        run.log(
            {
                str(id) : wandb.plot.line(
                    table, columns[0], columns[1], network_specific_title
                )
            }
        )
        chart_id += 1
    return chart_id

# global hyperparams
gamma = config["gamma"]
clip_epsilon = config["clip_epsilon"]
lr = config["lr"]
epochs = config["epochs"]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# logging arrays
losses_data : dict[str, list] = {} # lists are [x, y] data pairs, where y is loss and x is corresponding epoch 

rewards_data : dict[str, list] = {} # lists are [x, y] data pairs, where y is loss and x is corresponding epoch
reward_std_devs_data : dict[str, list] = {} # lists are [x, y] data pairs, where y is loss and x is corresponding epoch

timesteps_data : dict[str, list] = {} # lists are [x, y] data pairs, where y is loss and x is corresponding epoch
timestep_std_devs_data : dict[str, list] = {} # lists are [x, y] data pairs, where y is loss and x is corresponding epoch

def main():
    training_env_names = config["training_envs"]
    jumpstart_env_names = config["jumpstart_envs"]

    training_envs : list[gym.Env] = [gym.make(env, width=config["height"], height=config["width"]) for env in training_env_names]
    jumpstart_envs : list[gym.Env] = [gym.make(env, width=config["height"], height=config["width"]) for env in jumpstart_env_names]


    with wandb.init(project=config["project_name"], entity=config["wandb_entity"], config=config) as run:
        # ----- TRAIN ENVIRONMENT SPECIFIC AGENTS -----
        print("starting training env specific agents")
        env_specific_agents : dict[str, Agent] = {}
        for env_name, env in zip(training_env_names, training_envs):
            # input_dim = reduce(lambda x, y: x*y, env.observation_space['matrix'].shape) # this is VERY specific to my chessworld environments
            output_dim = env.action_space.n

            policy_net = PolicyNetwork(input_dim=config["height"]*config["width"], output_dim=output_dim).to(device=device)
            value_net = ValueNetwork(input_dim=config["height"]*config["width"]).to(device=device)
            policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
            value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
            
            # train policy
            policy_net = ppo_clip(policy_net=policy_net, value_net=value_net, 
                                policy_optimizer=policy_optimizer, value_optimizer=value_optimizer, 
                                envs=[env], epochs=epochs, run=run)

            agent : Agent = Agent(policy_network=policy_net, value_network=value_net, 
                                policy_optimizer=policy_optimizer, value_optimizer=value_optimizer)
        
            env_specific_agents[env_name] = agent
        
        # # ----- TRAIN MY AGENT -----
        # print("starting training my agent")
        # asa_policy = ASAgnosticPolicy(env_names=training_env_names, latent_action_dim=3, state_dim=config["height"]*config["width"]).to(device)
        # asa_value_net = ValueNetwork(input_dim=config["height"]*config["width"]).to(device) # same as env specific value net, is this an unfair comparison?
        # policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        # value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
        # idm = Encoder().to(device) 
        
        # # load trained IDM
        # idm.load_state_dict(torch.load("models/idm.pt"))
        
        # # freeze IDM params since it shouldn't be trained
        # for param in idm.parameters():
        #     param.requires_grad = False

        # # add action embeddings for each env
        # for env_name in training_env_names + jumpstart_env_names:
        #     action_embeddings = []

        #     env = gym.make(env_name, width=config["height"], height=config["width"])

        #     # use idm to make action embeddings. Add them in order of corresponding action indices.
        #     for i in range(env.action_space.n):
        #         s_t : np.ndarray = env.reset()[0]['matrix']
        #         s_tp1 : np.ndarray
        #         s_tp1 = env.step(i)[0]["matrix"]
        #         transition = torch.stack([torch.tensor(s_t), torch.tensor(s_tp1)]).to(torch.float32).to(device) # has shape (2, height, width)
        #         action_embeddings.append(idm(transition))

        #     asa_policy.Z_A[env_name] = action_embeddings

        # asa_policy = ppo_clip(policy_net=asa_policy, value_net=asa_value_net, 
        #                         policy_optimizer=policy_optimizer, value_optimizer=value_optimizer, 
        #                         envs=training_envs, epochs=epochs, run=run)
        
        # # asa_agent : Agent = Agent(policy_network=asa_policy, value_network=asa_value_net, 
        # #                                   policy_optimizer=policy_optimizer, value_optimizer=value_optimizer)

        # ----- EVAL JUMPSTART -----
        print("starting jumpstart evaluation")
        # train a policy and compare against trained action space agnostic 
        # policy. Use an env unseen by trained action space agnostic policy to see
        # policy's ability to transfer learning to a similar but unseen env. By 
        # comparing to initial performance of env specific policy, we can measure 
        # jumpstart. By comparing to fully trained env specific policy, we can provide 
        # a frame of reference for action space agnostic policy's 0-shot performances
        jumpstart_env_specific_agents : dict[str, Agent] = {}
        for jumpstart_env_name, jumpstart_env in zip(jumpstart_env_names, jumpstart_envs):
            # env_name = env.spec.id
            # train a jumpstart policy specific to each env
            output_dim = jumpstart_env.action_space.n

            jumpstart_eval_policy = PolicyNetwork(input_dim=config["height"]*config["width"], output_dim=output_dim).to(device=device)
            jumpstart_eval_value_net = ValueNetwork(input_dim=config["height"]*config["width"]).to(device=device)
            jumpstart_eval_policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
            jumpstart_eval_value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
            ppo_clip(policy_net=jumpstart_eval_policy, value_net=jumpstart_eval_value_net, 
                    policy_optimizer=jumpstart_eval_policy_optimizer, value_optimizer=jumpstart_eval_value_optimizer, 
                    envs=[jumpstart_env], epochs=epochs, run=run)
            agent : Agent = Agent(policy_network=jumpstart_eval_policy, value_network=jumpstart_eval_value_net, 
                                policy_optimizer=jumpstart_eval_policy_optimizer, value_optimizer=jumpstart_eval_value_optimizer)

            jumpstart_env_specific_agents[jumpstart_env_name] = agent
            evaluate(policy_net=asa_policy, policy_name="asa_policy", envs=[jumpstart_env], run=run, epoch=1)
        
        # log data
        id : int = 0
        window = 11
        
        # losses_data
        for network_name, data in losses_data.items():
            table = wandb.Table(data=data, columns=["epoch", "loss"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "loss", title=f"{network_name}'s loss at different epochs"
                    )
                }
            )
            id += 1
        
        # moving average of losses data
        moving_avg_losses : dict[str, list] = get_moving_avg(losses_data, window)
        for network_name, data in moving_avg_losses.items():
            table = wandb.Table(data=data, columns=["epoch", "loss"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "loss", title=f"moving average of {network_name}'s loss at different epochs"
                    )
                }
            )
            id += 1

        # rewards_data
        for network_name, data in rewards_data.items():
            table = wandb.Table(data=data, columns=["epoch", "avg_rewards"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "avg_rewards", title=f"{network_name}'s cumulative rewards averaged over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1

        # moving average of rewards_data
        moving_avg_rewards_data : dict[str, list] = get_moving_avg(rewards_data, window)
        for network_name, data in moving_avg_rewards_data.items():
            table = wandb.Table(data=data, columns=["epoch", "avg_rewards"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "avg_rewards", title=f"moving average of {network_name}'s cumulative rewards averaged over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1

        # reward_std_devs_data
        for network_name, data in reward_std_devs_data.items():
            table = wandb.Table(data=data, columns=["epoch", "rewards_std_dev"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "rewards_std_dev", title=f"standard deviation of {network_name}'s rewards over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1

        # moving average of reward_std_devs_data
        moving_avg_reward_std_devs_data : dict[str, list] = get_moving_avg(reward_std_devs_data, window)
        for network_name, data in moving_avg_reward_std_devs_data.items():
            table = wandb.Table(data=data, columns=["epoch", "rewards_std_dev"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "rewards_std_dev", title=f"moving average of standard deviation of {network_name}'s rewards over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1

        # timesteps_data
        for network_name, data in timesteps_data.items():
            table = wandb.Table(data=data, columns=["epoch", "avg_timesteps"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "avg_timesteps", title=f"{network_name}'s timesteps per rollout averaged over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1
        
        # moving average of timesteps_data
        moving_avg_timesteps_data : dict[str, list] = get_moving_avg(timesteps_data, window)
        for network_name, data in moving_avg_timesteps_data.items():
            table = wandb.Table(data=data, columns=["epoch", "avg_timesteps"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "avg_timesteps", title=f"moving average of {network_name}'s timesteps per rollout averaged over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1

        # timestep_std_devs_data
        for network_name, data in timestep_std_devs_data.items():
            table = wandb.Table(data=data, columns=["epoch", "timesteps_std_dev"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "timesteps_std_dev", title=f"standard deviation of {network_name}'s timesteps per rollout over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1

        # moving average of timestep_std_devs_data
        moving_avg_timestep_std_dev_data : dict[str, list] = get_moving_avg(timestep_std_devs_data, window)
        for network_name, data in moving_avg_timestep_std_dev_data.items():
            table = wandb.Table(data=data, columns=["epoch", "timesteps_std_dev"])
            run.log(
                {
                    str(id) : wandb.plot.line(
                        table, "epoch", "timesteps_std_dev", title=f"moving average of standard deviation of {network_name}'s timesteps per rollout over multiple test rollouts, at different epochs"
                    )
                }
            )
            id += 1

        print("finished!")


if __name__ == "__main__":
    main()

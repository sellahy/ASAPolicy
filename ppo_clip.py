from random import randrange
from math import floor, ceil
from collections import defaultdict
from statistics import mean
from typing import List, Dict, Any
import pickle
from torch.multiprocessing import Pool, set_start_method
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import chess_env
from idm_training import Encoder
import pdb

# Define the policy network
class PolicyNetwork(nn.Module):
    # param count guide:
    #  5x5 : param count 1989 - 2249
    # 10x10: param count 6789 - 7049
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs
 
# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class ASAPolicy(torch.nn.Module):
    """My action space agnostic (asa) policy"""
    # param count guide:
    #  5x5 : latent_action_dim=3, nhead=1, dim_feedforward=300 -> 2163 param count
    # 10x10: latent_action_dim=3, nhead=1, dim_feedforward=975 -> 6888 param count
    def __init__(self, idm : nn.Module, latent_action_dim, dim_feedforward, state_dim, nhead=1, num_layers=1):
        super().__init__()
        
        # freeze idm parameters since it shouldn't be trained
        for param in idm.parameters():
            param.requires_grad = False
        self.idm = idm

        assert latent_action_dim % nhead == 0, f"latent_action_dim must be divisible by nhead, but got latent_action_dim={latent_action_dim}, nhead={nhead}"

        self.state_dim = state_dim

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=latent_action_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers, enable_nested_tensor=False)
    
        self.state_embed = torch.nn.Linear(state_dim, latent_action_dim)

        self.Z_A : dict[str, list[torch.Tensor]] = {}
        self.current_env : str

    def make_action_embeddings(self, envs : list[gym.Env] | list[str]) -> None:
        """add action embeddings for each env"""
        
        # handle envs as name strings case 
        if type(envs[0]) == str: # assumes all elements of list are either gym.Envs or strings
            gym_envs : list[gym.Env] = [gym.make(env_name) for env_name in envs] # TODO: may need to add references to config here to include env arguments
        else:
            gym_envs = envs

        self.Z_A : dict[str, list[torch.Tensor]] = {env.unwrapped.spec.id.replace('chess_env/', '') : [] for env in gym_envs}
        self.current_env : str = gym_envs[0].unwrapped.spec.id.replace('chess_env/', '')

        for env in gym_envs:
            action_embeddings = []

            # VERY IMPORTANT: this is probably a massive area for improvement. I'm only using a single transition to make the latent actions
            # use idm to make action embeddings. Add them in order of corresponding action indices.
            for i in range(env.action_space.n):
                s_t : np.ndarray = env.reset()[0]#.flatten() idm expects unflattened states
                s_tp1 : np.ndarray = env.step(i)[0]#.flatten() idm expects unflattened states
                transition = torch.stack([torch.tensor(s_t), torch.tensor(s_tp1)]).to(torch.float32).to(device) # has shape (2, height, width)
                action_embeddings.append(self.idm(transition))

            self.Z_A[env.unwrapped.spec.id.replace('chess_env/', '')] = action_embeddings

    def forward(self, s) -> torch.Tensor:
        """Given a list of latent actions Z_A and a batch of states s, returns the transformer encoder output.
        Z_A: list of torch.Tensors representing latent actions corresponding to all actions in self.A
        s: tensor of shape (batch_size, state_size) representing a batch of states.
        Returns: torch.Tensor of shape (batch_size, latent_action_dim) representing the log probabilities of each action for each state in batch."""
        # assume Z_A is a list of latent actions (torch.Tensors) corresponding to all actions in self.A
        # assume s is a batch of states (torch.Tensor) of shape (batch_size, state_size)
        Z_A = self.Z_A[self.current_env]

        # expect input to be (batch_size, height*width). need to reshape input into correct shape
    
        s = s.clone().detach().to(torch.float32) # I forget the purpose of this line. I think it was just changing the dtype of the tensor, which could be done in a simpler way
        # s = torch.tensor(s, dtype=torch.float32)
        if s.dim() == 1:
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

# TODO: may need custom value network with transformer arch

def record_data(policy : nn.Module, epoch : int, env_name : str, 
                ep_return : float, ep_len : int, policy_loss : float, 
                value_net_loss : float) -> None:
    """record data for given policy"""
    assert policy in list(training_stats.keys()), f"attempted to record data for policy {policy} not found in training stats"
    
    training_stats[policy].append({"epoch" :          epoch,
                                   "env" :            env_name,
                                   "return" :         ep_return,
                                   "ep_len" :         ep_len,
                                   "policy_loss" :    policy_loss,
                                   "value_net_loss" : value_net_loss})

# def _smooth_data(array_to_smooth : list, window_size : int) -> list: # I made this for the bottom func, but might not be useful
#     """take an array and return the CMA of the array. For data near end points, 
#     where there isn't sufficient surrounding data to apply the full window, 
#     apply as much as possible. 
#     Ex. at 1st data point with window_size > 1, average that point with a half 
#     window's worth of data on the right"""
    
#     # compute CMA
#     smoothed_array = np.convolve(array_to_smooth, np.ones(window_size)/window_size, 'same')
    
#     # half the window_size (rounded down) data points on each side need fixing
#     for i in range(floor(window_size/2)):
#         smoothed_array[ i  ] = smoothed_array[  i ]*window_size/(ceil(window_size/2) + i)
#         smoothed_array[-i-1] = smoothed_array[-i-1]*window_size/(ceil(window_size/2) + i)
#     return smoothed_array

def _add_centered_moving_average(data: List[Dict[str, Any]],
                                key: str = "v",
                                window: int = 3,
                                out_key: str = "smoothed_v") -> None:
    """
    In-place: adds out_key to each dict in `data` as the centered moving average of `key`.
    Averages are computed separately per distinct env value (data is grouped by the dict's 'env' field).
    Window should be a positive odd or even integer; centering uses floor(window/2) on the left
    and ceil(window/2)-1 on the right (so boundary windows are truncated to available points).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    # Collect indices per env
    env_to_indices = defaultdict(list)
    for i, item in enumerate(data):
        env_to_indices[item.get("env")].append(i)

    half_left = window // 2
    half_right = window - half_left - 1  # ensures total length == window

    for env, indices in env_to_indices.items():
        # extract values for this env in the original order
        vals = [data[i][key] for i in indices]
        n = len(vals)
        # compute moving averages for each position j in vals
        for j, idx in enumerate(indices):
            start = max(0, j - half_left)
            end = min(n - 1, j + half_right)  # inclusive
            window_vals = vals[start:end + 1]
            avg = mean(window_vals) if window_vals else None
            data[idx][out_key] = avg

def insert_smoothed_data(training_stats_dicts : list[dict], window_size=5) -> None:
    """takes in a list of dictionaries containing training stats and smoothes 
    the training data by adding the centered moving average. The smoothed data 
    is added to the dictionaries and returned. The smoothed data is present 
    at all data points, but where insufficient data is available on each side 
    to calculate the CMA (i.e. everywhere but the first and last 
    floor(window_size/2) data points), a truncated window is used. For example, 
    the CMA on [3,5,8] with window size 3 centered at index 0 equals (3+5)/2. 
    Smoothes data that correponds to the same env (i.e. separate CMAs are 
    calculated for returns from env A and env B)."""


    _add_centered_moving_average(training_stats_dicts, key="return",         window=window_size, out_key="smoothed_return")
    

    _add_centered_moving_average(training_stats_dicts, key="ep_len",         window=window_size, out_key="smoothed_ep_len")
    
    
    _add_centered_moving_average(training_stats_dicts, key="policy_loss",    window=window_size, out_key="smoothed_policy_loss")
    
    
    _add_centered_moving_average(training_stats_dicts, key="value_net_loss", window=window_size, out_key="smoothed_value_net_loss")

    # # Centered moving average starts after there is enough data in the window 
    # # to calculate the CMA
    # idx_start = floor(window_size/2)
    # idx_end = len(training_stats_dicts) - floor(window_size/2) - 1

    # # for i in range(idx_start, idx_end+1):
    # for i in range(len(training_stats_dicts)):
    #     # smooth data that correponds to the same env
    #     returns_to_smooth =          [training_stats_dicts[j][    "return"    ] for j in range(len(training_stats_dicts)) if training_stats_dicts[j]["env"]==training_stats_dicts[i]["env"]]
    #     training_stats_dicts[i]["smoothed_return"] =         sum([returns_to_smooth[j]          for j in range(i-floor(window_size/2), i+floor(window_size/2)+1)])/window_size
        
    #     ep_lens_to_smooth =          [training_stats_dicts[j][    "ep_len"   ] for j in range(len(training_stats_dicts)) if training_stats_dicts[j]["env"]==training_stats_dicts[i]["env"]]
    #     training_stats_dicts[i]["smoothed_ep_len"] =         sum([ep_lens_to_smooth[j]          for j in range(i-floor(window_size/2), i+floor(window_size/2)+1)])/window_size
        
    #     policy_losses_to_smooth =    [training_stats_dicts[j][ "policy_loss"  ] for j in range(len(training_stats_dicts)) if training_stats_dicts[j]["env"]==training_stats_dicts[i]["env"]]
    #     training_stats_dicts[i]["smoothed_policy_loss"] =    sum([policy_losses_to_smooth[j]    for j in range(i-floor(window_size/2), i+floor(window_size/2)+1)])/window_size
        
    #     value_net_losses_to_smooth = [training_stats_dicts[j]["value_net_loss"] for j in range(len(training_stats_dicts)) if training_stats_dicts[j]["env"]==training_stats_dicts[i]["env"]]
    #     training_stats_dicts[i]["smoothed_value_net_loss"] = sum([value_net_losses_to_smooth[j] for j in range(i-floor(window_size/2), i+floor(window_size/2)+1)])/window_size

def training_setup(agents : list[nn.Module]):
    for agent in agents:
        training_stats[agent] = []

def _get_env_of_agent(agent : nn.Module) -> str:
    """returns the name of the env that the agent passed as argument was 
    trained in. If agent is asa_agent, this function will return the first
    environment it saw (which means this function doesn't work for it)"""
    return training_stats[agent][0]["env"]

# PPO - Clip implementation
def ppo_clip(policy_net : nn.Module, value_net : nn.Module, 
             policy_optimizer, value_optimizer, 
             envs : list[gym.Env]):
    print(f"starting ppo_clip for {[env.unwrapped.spec.id.replace('chess_env/', '') for env in envs]}")
    timesteps_taken = 0
    epoch = 0
    # input_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
    # output_dim = env.action_space.n

    # policy_net = PolicyNetwork(input_dim, output_dim)
    # value_net = ValueNetwork(input_dim)
    # policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    # for epoch in range(epochs):
    while timesteps_taken < total_timesteps:
        env = envs[randrange(len(envs))]
        try:
            policy_net.current_env = env.unwrapped.spec.id.replace('chess_env/', '').replace('chess_env/', '')
        except AttributeError:
            print("continuing ppo with simple policy")
        states, actions, rewards, log_probs_old = [], [], [], []
        state = env.reset()
        state = state[0].flatten()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device=device)
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = next_state.flatten()
            done = terminated or truncated
 
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
 
        states =        torch.FloatTensor(np.array(states)).to(device) # convert to numpy array first to silence warning about slowness of converting list of np.ndarrays -> torch.Tensor
        actions =       torch.LongTensor(np.array(actions)).to(device) # convert to numpy array first to silence warning about slowness of converting list of np.ndarrays -> torch.Tensor
        log_probs_old = torch.FloatTensor(np.array(log_probs_old)).to(device) # convert to numpy array first to silence warning about slowness of converting list of np.ndarrays -> torch.Tensor
 
        # Calculate advantages
        values = value_net(states).squeeze()
        advantages = returns - values.detach()
 
        # Update policy
        action_probs = policy_net(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        ratios = torch.exp(log_probs - log_probs_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
 
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
 
        # Update value function
        value_loss = nn.MSELoss()(values, returns) # complains about shape mismatch, but printing shapes indicates they are the same shape. This line is still functional regardless
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # print(f"timesteps taken: {timesteps_taken}, epoch: {epoch}, episodic return: {sum(rewards)}, episode length: {len(rewards)}")
        timesteps_taken += len(rewards)
        epoch += 1

        # eval once every eval_frequency epochs
        if epoch % eval_frequency == 0:
            for environment in envs:
                ep_return, ep_len = evaluate(policy_net, environment)
                record_data(policy_net, 
                            epoch, 
                            environment.unwrapped.spec.id.replace('chess_env/', ''), 
                            ep_return, 
                            ep_len, 
                            policy_loss.item(), 
                            value_loss.item())
            
 
    # eval one last time after training is done
    for environment in envs:
        ep_return, ep_len = evaluate(policy_net, environment)
        record_data(policy_net, 
                    epoch, 
                    environment.unwrapped.spec.id.replace('chess_env/', ''), 
                    ep_return, 
                    ep_len, 
                    policy_loss.item(), 
                    value_loss.item())

    # smooth data before returning from training loop
    insert_smoothed_data(training_stats[policy_net])

    # return policy_net, value_net, epoch
    return epoch

def evaluate(policy : nn.Module, env : gym.Env) -> tuple:
    try:
        policy.current_env = env.unwrapped.spec.id.replace('chess_env/', '')
    except AttributeError:
        print("continuing eval with simple policy")
    total_rewards = []
    episode_lengths = []
    for _ in range(100):
        rewards = []
        state = env.reset()
        state = state[0].flatten()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = policy(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = next_state.flatten()
            done = terminated or truncated

            rewards.append(reward)

            state = next_state
        total_rewards.append(sum(rewards))
        episode_lengths.append(len(rewards))
    avg_return = sum(total_rewards)/len(total_rewards)
    avg_ep_len = sum(episode_lengths)/len(episode_lengths)
    return avg_return, avg_ep_len

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# global hyperparams
gamma = 0.99
clip_epsilon = 0.2
lr = 0.001
total_timesteps = 3
eval_frequency = 2 # number of episodes between evaluations within training loop

# keys are policies. For each policy, there is a list of data points stored 
# as dictionaries (ex. {"epoch" : 20, "env" : KingWorld-v0, "return" : 20, "ep_len" : 10, "policy_loss" : 200, "value_net_loss" : 100})
training_stats : dict[nn.Module,list[dict]] = {} 

if __name__ == '__main__':
    print(f"training for {total_timesteps} timesteps")
    envs = [gym.make("chess_env/ChessWorld-v0"),
            gym.make("chess_env/BishopWorld-v0"),
            gym.make("chess_env/CamelWorld-v0"),
            gym.make("chess_env/GoldGeneralWorld-v0"),
            gym.make("chess_env/KingWorld-v0"),
            gym.make("chess_env/KnightWorld-v0"),
            gym.make("chess_env/SilverGeneralWorld-v0"),
            gym.make("chess_env/ZebraWorld-v0")]

    # initialize policy nets, value nets, and optimizers. Location corresponds to environments above
    policy_nets = [PolicyNetwork(env.observation_space.shape[0]*env.observation_space.shape[1], env.action_space.n).to(device) for env in envs]
    value_nets = [ValueNetwork(env.observation_space.shape[0]*env.observation_space.shape[1]).to(device) for env in envs]
    policy_optimizers = [optim.Adam(policy.parameters(), lr=lr) for policy in policy_nets]
    value_optimizers = [optim.Adam(value_net.parameters(), lr=lr) for value_net in value_nets]


    # load trained IDM
    idm = Encoder().to(device)    
    idm.load_state_dict(torch.load("models/idm.pt", weights_only=False))

    # freeze IDM params since it shouldn't be trained
    for param in idm.parameters():
        param.requires_grad = False

    # initialize action space agnostic policy, value net, and optimizers
    asa_policy = ASAPolicy(idm, latent_action_dim=3, dim_feedforward=300, state_dim=envs[0].observation_space.shape[0]*envs[0].observation_space.shape[1]).to(device)
    asa_policy.make_action_embeddings(envs)
    asa_value_net = ValueNetwork(envs[0].observation_space.shape[0]*envs[0].observation_space.shape[1]).to(device)
    policy_optimizer = optim.Adam(asa_policy.parameters(), lr=lr)
    value_optimizer = optim.Adam(asa_value_net.parameters(), lr=lr)
    asa_envs = envs[0:4]

    # set up training
    training_setup(policy_nets + [asa_policy])

    # print stats
    set_start_method('spawn')
    # with Pool() as p:
    #     out = p.starmap(ppo_clip, [(policy_nets[i],       value_nets[i],
    #                                 policy_optimizers[i], value_optimizers[i],
    #                                 [envs[i]]) for i in range(len(envs))]
    #                               + [(asa_policy, asa_value_net, policy_optimizer, value_optimizer, asa_envs)]) # train asa agent on first 4 envs
    #     # out = [(asa_policy, asa_value_net, policy_optimizer, value_optimizer, asa_envs)]
    out = [ppo_clip(policy_nets[i],       value_nets[i],
                    policy_optimizers[i], value_optimizers[i],
                    [envs[i]]) for i in range(len(envs))] + [ppo_clip(asa_policy, asa_value_net, policy_optimizer, value_optimizer, asa_envs)]
    print(f"total timesteps: {total_timesteps}")

    epochs = out
    epochs = [(envs[i].unwrapped.spec.id.replace('chess_env/', ''), epochs[i]+1) for i in range(len(envs))] # QUESTION: why did I add the +1?
    print(f"environments and their corresponding epochs: {epochs}")

    # env specific agent evaluations
    with Pool() as p:
        # evals = p.starmap(evaluate, [(out[i][0], envs[i]) for i in range(len(envs))])
        evals = p.starmap(evaluate, [(policy_nets[i], envs[i]) for i in range(len(envs))])
    # evals = [evaluate(out[i][0], envs[i]) for i in range(len(envs))]
    # evals is a list of tuples. Each has 2 items, avg return and len. I want to print a list of tuples with 3 items - env name, avg return, avg len
    envs_and_evals = {envs[i].unwrapped.spec.id.replace('chess_env/', '') : evals[i] for i in range(len(envs))}
    print(f"envs eval: {envs_and_evals}")

    # asa policy evaluation within envs that it was trained on
    with Pool() as p:
        asa_evals = p.starmap(evaluate, [(asa_policy, asa_env) for asa_env in asa_envs])
    asa_envs_and_evals = {asa_envs[i].unwrapped.spec.id.replace('chess_env/', '') : evals[i] for i in range(len(asa_envs))}
    print(f"asa_envs eval: {asa_envs_and_evals}")

    # TODO: compare training speed. How should I use the recorded training data to compare training speed? I can now record training stats and open them in graphs using matplotlib
    # TODO: compare asymptotic performance. However, this is likely meaningless in this environment, since the agents achieve the max reward

    # jumpstart evaluation
    # jumpstart must be evaluated on an agent with another environment with 
    # the same action space as its training environment. evaluate jumpstart
    # between following sets of agents:
    # BishopWorld, ChessWorld
    # CamelWorld, KingWorld, KnightWorld, ZebraWorld
    # NOT GoldGeneralWorld (7 actions) or SilverGeneralWorld (6 actions)
    # inputs is tuples of agents and the environments to evaluate them in
    inputs = [(policy_nets[i], envs[j]) # env specific agents first
              for i in range(len(policy_nets)) 
              for j in range(len(envs)) 
              if i != j and envs[i].action_space.n == envs[j].action_space.n] + \
              [(asa_policy, env) for env in envs if env not in asa_envs] # then asa agent
    
    # remove agents that have no other environments to train in
    inputs = [input_tuple for input_tuple in inputs if input_tuple[1]!=[]] 
    
    with Pool() as p:
        jumpstart_out = p.starmap(evaluate, inputs) 
    # dictionary with tuples of agents and the environments they were evaluated 
    # in as keys and tuples of avg_return and avg_ep_len as values
    jumpstart_evals = dict(zip(inputs, jumpstart_out)) 

    # save models
    for model in list(training_stats.keys()):
        is_asa_agent = False if len(set([datum["env"] for datum in training_stats[model]])) == 1 else True # assumes asa agent will train in multiple envs
        model_name =  "asa_agent" if is_asa_agent == True else f"{training_stats[model][0]['env']}_agent"
        torch.save(model, f"save_data/models/{model_name}")

    # save data
    with open("save_data/training_stats.pickle", "wb") as f:
        pickle.dump(training_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("save_data/jumpstart_eval.pickle", "wb") as f:
        pickle.dump(jumpstart_evals, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("done!")

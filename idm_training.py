from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb
import gymnasium as gym
import chess_env
from config import idm_training_config_dict as config
from math import sqrt
from random import randint

def custom_collate_fn(batch):
    """returns batch as a dict with env names as keys and data tensors as values. Data tensors have shape (batch_size, 2, height, width)"""
    # batch is a list of datums from __getitem__()
    collated_batch : dict[str, torch.Tensor] = {}
    
    # add each torch tensor to its respective data tensor value in collated_batch
    for datum in batch:
        env_name : str = datum[1]
        datum_tensor : torch.Tensor = datum[0] # expect this to have shape (1,2,height,width) so we can use torch.cat along first dim later

        if env_name not in list(collated_batch.keys()):
            # intialize data tensor
            collated_batch[env_name] = datum_tensor 
        else:
            # add datum tensor to existing data tensor
            collated_batch[env_name] = torch.cat([collated_batch[env_name], datum[0]])

    return collated_batch

class ChessDataset(Dataset):
    def __init__(self, gym_envs : list[gym.Env]) -> None:
        super().__init__()

        # to get dataset started, get one datum
        random_env = gym_envs[randint(0, len(gym_envs)-1)]
        self.dataset : list[tuple] = [self._get_datum(random_env)]

        # create dataset by repeated sampling an environment and 
        # then adding a pair of consecutive observations (a datum) to the dataset
        while len(self.dataset) <= config["dataset_size"]:
            random_env = gym_envs[randint(0, len(gym_envs)-1)]
            self.dataset.append(self._get_datum(random_env))

    def __len__(self):
        # return batch size dimension of dataset
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        return self.dataset[index]
    
    def _get_datum(self, env : gym.Env) -> tuple[torch.Tensor, str]:
        """get datum in shape (1, 2, height, width), where first dim represents
        batch size (1 bc this method gets only a single datum), 2nd dim 
        represents a stack of 2 consecutive observations, 3rd dim represents
        height of observation grid, and 4th dim represents width of observation grid"""
        # get num of actions in env for sake of generating a random action
        try:
            actions_in_env : int = env.action_space.n
        except EnvironmentError:
            assert False, f"expected env {env} to be discrete, but got something else"
        
        random_action_idx = randint(0, actions_in_env-1)
        
        # handle cases where env needs to be reset before/after first action
        try:
            s_t = env.step(random_action_idx)
        except gym.error.ResetNeeded:
            # handle environment not having been reset yet
            env.reset()
            return self._get_datum(env)
        else:
            # if env finishes after first step, we can't get the data we need, 
            # as we need 2 consecutive timesteps. If this happens, reset and try again.
            terminated = s_t[2]
            truncated = s_t[3]
            if terminated == True or truncated == True:
                env.reset()
                return self._get_datum(env)

        random_action_idx = randint(0, actions_in_env-1)

        s_tp1 = env.step(random_action_idx)

        obs_at_s_t = s_t[0]["matrix"] # numpy array with shape (height, width)
        obs_at_s_tp1 = s_tp1[0]["matrix"] # numpy array with shape (height, width)
        datum = torch.stack([torch.tensor(obs_at_s_t, dtype=torch.float32), torch.tensor(obs_at_s_tp1, dtype=torch.float32)]) # # has shape (2, height, width)
        return (torch.unsqueeze(datum, 0), env.spec.id) # has shape (1, 2, height, width)

class Encoder(nn.Module):
    def __init__(self, input_dim=config["height"]*config["width"], 
                 latent_dim=3): # for chessworld, latent dim should be < 4 because transition can be exactly determined by 4 numbers
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim # width by height
        self.latent_dim = latent_dim # dimension of latent embedding

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2*input_dim, 40), # 2*input_dim because input is a flattened _pair_ of observations
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, latent_dim)
        )

    def forward(self, x):
        """Expect input with shape (batch_size, 2, height, width). Output will have shape (batch_size, latent_dim)"""
        assert x.shape[x.dim()-3:x.dim()] == (2, int(sqrt(self.input_dim)), int(sqrt(self.input_dim))), f"expected input to have shape (batch_size, 2, {int(sqrt(self.input_dim))}, {int(sqrt(self.input_dim))}), but got {x.shape}"
        
        # add explicit batch dim if not present
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        batch_size : int = x.shape[0]

        # Flatten input
        x = x.view(batch_size, -1,) # has shape (batch_size, 2*height*width)

        # Get latent representation
        latent = self.encoder(x) # has shape (batch_size, 3)

        assert latent.shape == (batch_size, self.latent_dim), f"expected input with shape ({batch_size}, {self.latent_dim}), but got {latent.shape}"
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim=3,  # for chessworld, latent dim should be < 4 because transition can be exactly determined by 4 numbers
                 output_dim=config["height"]*config["width"]):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim # dimension of latent embedding
        self.output_dim = output_dim # width by height

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, output_dim*2), # 2*output_dim because output is a flattened _pair_ of observations
            nn.Tanh()
        )

    def forward(self, latent):
        """Expect input with shape (batch_size, latent_dim). Output will have shape (batch_size, 2, height, width)"""
        assert latent.shape[1:] == torch.Size([self.latent_dim]), f"expected input with shape (batch_size, {self.latent_dim}), but got {latent.shape}"
        batch_size : int = latent.shape[0]

        # Flatten input
        latent = latent.view(latent.size(0), -1)

        # Reconstruct input
        reconstructed = self.decoder(latent)

        # Reshape to original dimensions
        reconstructed = reconstructed.view(-1, 2, int(sqrt(self.output_dim)), int(sqrt(self.output_dim)))
        assert reconstructed.shape == (batch_size, 2, int(sqrt(self.output_dim)), int(sqrt(self.output_dim))), f"expected reconstructed to have shape (batch_size, 2, height, width) = ({batch_size}, 2, {int(sqrt(self.output_dim))}, {int(sqrt(self.output_dim))}), but got {reconstructed.shape}"
        return reconstructed

# train IDM in autoencoder fashion for every environment in training environment. 
# each environment will share an encoder and train a decoder from scratch
# initialize encoder
# intialize decoder for each environment
# 
# For every env in training envs, train encoder-decoder on transition tuples. 
# Tuples need not come from an expert policy, can come from a random policy.
# 
# Train stack by reconstruction loss to reconstruct (s_t, s_t+1) given (s_t, s_t+1). 
# ISSUE: However, this won't work with providing IDM multiple SAS tuples since it will only produce one
# alternatively, could reconstruct pixelwise difference between states. This makes 
# sense for very static environments, but maybe less so for very dynamic environments. 
# Still doesn't handle aformentioned issue
#
# to start, environments will be variations of gridworld with different movement patterns. 
# Ex. knight movement pattern, 
# king movement pattern, 
# rook (1 in cardinal directions), 
# bishop (1 in ordinal directions), 
# gold general (same as king, but missing bottom left and bottom right tiles), 
# silver general (same as king but missing left, right, and down), 
# dragon king (as many tiles as possible in cardinal directions, 1 in ordinal directions), 
# dragon horse (as many tiles as possible in ordinal directions, 1 in cardinal directions)
# camel (2 tiles in a cardinal direction, 1 tile diagonally outward)
# zebra (3 tiles in cardinal direction, then 2 orthogonally)


# loss function is BCE (since states are 0's where agent is not and a 1 where agent is), input s_t and s_t+1 and reconstruct it
# TODO: make script that trains IDM then policy and records performance stats. Include hyperparam sweeps.


def main():
    project_name : str = "simple_idm"

    torch.manual_seed(config["seed"])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    training_envs : list[gym.Env] = [gym.make(env_name, width=config["height"], height=config["width"]) for env_name in config["training_envs"]]

    train_dataset = ChessDataset(training_envs)
    
    # each datum is a tensor with shape (2,width,height)
    # corresponding to a stack of s_t and s_t+1, where each is a 2d tensor with 
    # shape (width,height)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    # initalize IDM. This will be common to all environments
    idm = Encoder().to(device)

    decoders : dict[str, nn.Module] = {}
    # initialize decoders. These will be environment specific.
    for env in training_envs:
        decoders[env.spec.id] = Decoder().to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(idm.parameters()) + [param for decoder in decoders.values() for param in decoder.parameters()], lr=config["lr"])

    # initialize wandb run
    with wandb.init(project=config["project_name"], entity=config["wandb_entity"], config=config) as run:
        # training loop
        num_epochs = config["epochs"]
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, data_dict in enumerate(train_loader):
                # data_dict is a dict with env names as keys and data tensors as values

                intrabatch_losses = []

                # data is heterogeneous. Iterate through data for each env and perform forward and backwards pass.
                for env_name in list(data_dict.keys()):
                    current_decoder = decoders[env_name]

                    # data tensor with shape (intrabatch_size, 2, height, width)
                    # intrabatch_size is random and between 1 and Dataloader's batch size
                    x = data_dict[env_name]

                    # Move data to device
                    x = x.to(device)

                    # Forward pass
                    outputs = current_decoder(idm(x))
                    loss = criterion(outputs, x)
                    intrabatch_losses.append(loss)

                # aggregate losses
                batch_loss = torch.stack(intrabatch_losses).sum()

                # Backward pass and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
            
                # data logging and saving
                if batch_idx % 5 == 0:
                    # save models so far, for purposes of use with policy or resuming training
                    torch.save(idm.state_dict(), "models/idm.pt")
                    for env_spec_id, decoder in decoders.items():
                        # exclude all slashes otherwise torch.save tries to save inside chess_env folder
                        # use only part after last slash as decoder name
                        decoder_name : str = env_spec_id[str.rfind(env_spec_id, '/')+1:] 
                        torch.save(decoder.state_dict(), f"models/{decoder_name}_decoder.pt")

                    # save optimizer state for resuming training
                    torch.save(optimizer.state_dict(), "models/optimizer.pt")

                    # record training stats
                    run.log({"batch_loss" : batch_loss.item(), "epoch" : epoch, "batch" : batch_idx})
                    
                    for i, env_name in enumerate(list(data_dict.keys())):
                        run.log({f"{env_name}_loss" : intrabatch_losses[i], "epoch" : epoch, "batch" : batch_idx})
            
            # log epoch statistics
            run.log({"epoch_loss" : total_loss, "epoch" : epoch})

            avg_loss = total_loss / len(train_loader)
            run.log({"epoch_loss_averaged_over_batches" : avg_loss, "epoch" : epoch})
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main()
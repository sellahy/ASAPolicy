from typing import Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb
import gymnasium as gym
import chess_env
from config import idm_training_config_dict as config # FIXME: may want to remove this just to make sure IDM and decoder use the correct config. Current use of cfg because config isn't available is inconsistent with ppo_clip.py
from math import sqrt
from random import randint

def custom_collate_fn(batch: list) -> dict[str, torch.Tensor]:
    """Returns batch as a dict with env names as keys and data tensors as values.

    Data tensors have shape (batch_size, 2, height, width).
    """
    # batch is a list of datums from __getitem__()
    collated_batch: dict[str, torch.Tensor] = {}

    # add each torch tensor to its respective data tensor value in collated_batch
    for datum in batch:
        env_name: str = datum[1]
        datum_tensor: torch.Tensor = datum[0]  # expect shape (1, 2, height, width)

        if env_name not in list(collated_batch.keys()):
            # intialize data tensor
            collated_batch[env_name] = datum_tensor
        else:
            # add datum tensor to existing data tensor
            collated_batch[env_name] = torch.cat([collated_batch[env_name], datum[0]])

    return collated_batch

class ChessDataset(Dataset):
    def __init__(self, gym_envs: list[gym.Env], dataset_size: int) -> None:
        super().__init__()

        # to get dataset started, get one datum
        random_env: gym.Env = gym_envs[randint(0, len(gym_envs)-1)]
        self.dataset: list[tuple] = [self._get_datum(random_env)]

        # create dataset by repeated sampling an environment and
        # then adding a pair of consecutive observations (a datum) to the dataset
        while len(self.dataset) <= dataset_size:
            random_env = gym_envs[randint(0, len(gym_envs)-1)]
            self.dataset.append(self._get_datum(random_env))

    def __len__(self) -> int:
        # return batch size dimension of dataset
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def _get_datum(self, env: gym.Env) -> tuple[torch.Tensor, str]:
        """Get datum in shape (1, 2, height, width).

        First dim is batch size (1, since this method gets a single datum),
        2nd dim is a stack of 2 consecutive observations, 3rd dim is height,
        and 4th dim is width.
        """
        # get num of actions in env for sake of generating a random action
        try:
            actions_in_env: int = env.action_space.n
        except EnvironmentError:
            assert False, f"expected env {env} to be discrete, but got something else"

        random_action_idx: int = randint(0, actions_in_env-1)

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
            terminated: bool = s_t[2]
            truncated: bool = s_t[3]
            if terminated == True or truncated == True:
                env.reset()
                return self._get_datum(env)

        random_action_idx = randint(0, actions_in_env-1)
        s_tp1 = env.step(random_action_idx)

        # s_t[0] and s_tp1[0] are flat 2D numpy arrays (height, width) — the
        # observation space returns the grid matrix directly (not a dict).
        obs_at_s_t: torch.Tensor = torch.tensor(s_t[0], dtype=torch.float32)
        obs_at_s_tp1: torch.Tensor = torch.tensor(s_tp1[0], dtype=torch.float32)
        datum: torch.Tensor = torch.stack([obs_at_s_t, obs_at_s_tp1])  # shape (2, height, width)
        return (torch.unsqueeze(datum, 0), env.spec.id)  # shape (1, 2, height, width)

class Encoder(nn.Module):
    def __init__(self, height : int = config["height"], width : int = config["width"], 
                 input_dim: int = config["height"]*config["width"], latent_dim: int = 3):  # for chessworld envs, latent dim should be < 4 because transition can be exactly determined by 4 numbers
        super(Encoder, self).__init__()

        self.height = height
        self.width = width
        self.input_dim: int = input_dim   # width by height
        self.latent_dim: int = latent_dim  # dimension of latent embedding

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2*input_dim, 40),  # 2*input_dim because input is a flattened _pair_ of observations
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expect input with shape (batch_size, 2, height, width). Output will have shape (batch_size, latent_dim)"""
        assert x.shape[x.dim()-3:x.dim()] == (2, self.height, self.width), f"expected input to have shape (batch_size, 2, {self.height}, {self.width}), but got {x.shape}"

        # add explicit batch dim if not present
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size: int = x.shape[0]

        # Flatten input
        x = x.view(batch_size, -1,)  # has shape (batch_size, 2*height*width)

        # Get latent representation
        latent: torch.Tensor = self.encoder(x)  # has shape (batch_size, latent_dim)

        assert latent.shape == (batch_size, self.latent_dim), f"expected input with shape ({batch_size}, {self.latent_dim}), but got {latent.shape}"
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 3, # for chessworld, latent dim should be < 4 because transition can be exactly determined by 4 numbers
                 height : int = config["height"], width : int = config["width"], 
                 output_dim: int = config["height"]*config["width"]):
        super(Decoder, self).__init__()

        self.height = height
        self.width = width
        self.latent_dim: int = latent_dim   # dimension of latent embedding
        self.output_dim: int = output_dim   # width by height

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, output_dim*2),  # 2*output_dim because output is a flattened _pair_ of observations
            nn.Tanh()
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Expect input with shape (batch_size, latent_dim). Output will have shape (batch_size, 2, height, width)"""
        assert latent.shape[1:] == torch.Size([self.latent_dim]), f"expected input with shape (batch_size, {self.latent_dim}), but got {latent.shape}"
        batch_size: int = latent.shape[0]

        # Flatten input
        latent = latent.view(latent.size(0), -1)

        # Reconstruct input
        reconstructed: torch.Tensor = self.decoder(latent)

        # Reshape to original dimensions
        reconstructed = reconstructed.view(-1, 2, self.height, self.width)
        assert reconstructed.shape == (batch_size, 2, self.height, self.width), f"expected reconstructed to have shape (batch_size, 2, height, width) = ({batch_size}, 2, {self.height}, {self.width}), but got {reconstructed.shape}"
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


def main(cfg: dict = config, run_dir: Path = Path("models")) -> None:
    """Train the IDM encoder and per-environment decoders.

    Args:
        cfg:     Full config dict. Defaults to the module-level idm_training_config_dict
                 for backward-compatible standalone usage.
        run_dir: Directory to save model checkpoints. Created if it doesn't exist.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])

    # Device configuration
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_envs: list[gym.Env] = [
        gym.make(env_name, width=cfg["width"], height=cfg["height"])
        for env_name in cfg["training_envs"]
    ]

    train_dataset: ChessDataset = ChessDataset(training_envs, cfg["dataset_size"])

    # each datum is a tensor with shape (2, width, height)
    # corresponding to a stack of s_t and s_t+1, where each is a 2d tensor with
    # shape (width, height)
    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    # initalize IDM. This will be common to all environments
    input_dim: int = cfg["height"] * cfg["width"]
    idm: Encoder = Encoder(
        height=cfg["height"],
        width=cfg["width"],
        input_dim=input_dim,
        latent_dim=cfg["idm_latent_dim"]
    ).to(device)

    decoders: dict[str, nn.Module] = {}
    # initialize decoders. These will be environment specific.
    for env in training_envs:
        decoders[env.spec.id] = Decoder(
            latent_dim=cfg["idm_latent_dim"],
            height=cfg["height"],
            width=cfg["width"],
            output_dim=input_dim
        ).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(idm.parameters()) + [param for decoder in decoders.values() for param in decoder.parameters()],
        lr=cfg["lr"]
    )

    # initialize wandb run
    with wandb.init(project=cfg["project_name"], entity=cfg["wandb_entity"], config=cfg) as run:
        # training loop
        num_epochs: int = cfg["epochs"]
        for epoch in range(num_epochs):
            total_loss: float = 0.0
            for batch_idx, data_dict in enumerate(train_loader):
                # data_dict is a dict with env names as keys and data tensors as values

                intrabatch_losses: list[torch.Tensor] = []

                # data is heterogeneous. Iterate through data for each env and perform forward and backwards pass.
                for env_name in list(data_dict.keys()):
                    current_decoder: nn.Module = decoders[env_name]

                    # data tensor with shape (intrabatch_size, 2, height, width)
                    # intrabatch_size is random and between 1 and Dataloader's batch size
                    x: torch.Tensor = data_dict[env_name]

                    # Move data to device
                    x = x.to(device)

                    # Forward pass
                    outputs: torch.Tensor = current_decoder(idm(x))
                    loss: torch.Tensor = criterion(outputs, x)
                    intrabatch_losses.append(loss)

                # aggregate losses
                batch_loss: torch.Tensor = torch.stack(intrabatch_losses).sum()

                # Backward pass and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

                # data logging and saving
                if batch_idx % 5 == 0:
                    # save models so far, for purposes of use with policy or resuming training
                    torch.save(idm.state_dict(), run_dir / "idm.pt")
                    for env_spec_id, decoder in decoders.items():
                        # exclude all slashes otherwise torch.save tries to save inside chess_env folder
                        # use only part after last slash as decoder name
                        decoder_name: str = env_spec_id[str.rfind(env_spec_id, '/')+1:]
                        torch.save(decoder.state_dict(), run_dir / f"{decoder_name}_decoder.pt")

                    # save optimizer state for resuming training
                    torch.save(optimizer.state_dict(), run_dir / "optimizer.pt")

                    # record training stats
                    run.log({"batch_loss": batch_loss.item(), "epoch": epoch, "batch": batch_idx})

                    for i, env_name in enumerate(list(data_dict.keys())):
                        run.log({f"{env_name}_loss": intrabatch_losses[i], "epoch": epoch, "batch": batch_idx})

            # log epoch statistics
            run.log({"epoch_loss": total_loss, "epoch": epoch})

            avg_loss: float = total_loss / len(train_loader)
            run.log({"epoch_loss_averaged_over_batches": avg_loss, "epoch": epoch})
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main()

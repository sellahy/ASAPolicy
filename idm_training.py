import itertools
from typing import Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import wandb
import gymnasium as gym
import chess_env
from config import idm_training_config_dict as config # FIXME: may want to remove this just to make sure IDM and decoder use the correct config. Current use of cfg because config isn't available is inconsistent with ppo_clip.py


# ---------------------------------------------------------------------------
# Module-level helpers (shared with ppo_clip.py)
# ---------------------------------------------------------------------------

def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """Convert a raw observation ndarray to (C, H, W) float32 in [0, 1].

    Chess envs return (H, W) int arrays with values in {0, 1, 2}.
    Robot envs return (H, W, 3) uint8 arrays with values in [0, 255].
    Output is always (C, H, W) float32.  # OBS_CHANNELS_HOOK
    """
    if obs.ndim == 2:
        # Chess grid: (H, W) → (1, H, W), normalise 0–2 → 0–1
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0) / 2.0
    # RGB image: (H, W, 3) → (3, H, W), normalise 0–255 → 0–1
    return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0


def collect_n_transitions(
    env: gym.Env,
    action: int,
    n_transitions: int,
) -> torch.Tensor:
    """Collect N non-terminal (s_t, s_t+1) pairs for a single action.

    Each starting state s_t is drawn by calling env.reset(), so the N
    transitions are independent samples from across the state space rather
    than a single trajectory.

    Args:
        env:           Gymnasium environment (must support discrete actions).
        action:        Action index to execute at each starting state.
        n_transitions: Number of non-terminal transitions to collect.

    Returns:
        Tensor of shape (N, 2, C, H, W).

    Raises:
        RuntimeError: if n_transitions * 20 attempts are exhausted before N
                      non-terminal transitions are collected. This most likely
                      means the action terminates the episode from nearly every
                      starting state.
    """
    transitions: list[torch.Tensor] = []
    max_attempts: int = n_transitions * 20

    for _ in range(max_attempts):
        if len(transitions) == n_transitions:
            break
        try:
            s_t, _ = env.reset()
        except Exception:
            continue
        try:
            s_tp1, _, terminated, truncated, _ = env.step(action)
        except gym.error.ResetNeeded:
            s_t, _ = env.reset()
            s_tp1, _, terminated, truncated, _ = env.step(action)

        transitions.append(torch.stack([obs_to_tensor(s_t), obs_to_tensor(s_tp1)]))

    if len(transitions) < n_transitions:
        raise RuntimeError(
            f"collect_n_transitions: could not collect {n_transitions} non-terminal "
            f"transitions for action {action} in env '{env.spec.id}' after "
            f"{max_attempts} attempts."
        )

    return torch.stack(transitions)  # (N, 2, C, H, W)


# ---------------------------------------------------------------------------
# Dataset and collate
# ---------------------------------------------------------------------------

def custom_collate_fn(
    batch: list[tuple[torch.Tensor, str]]
) -> tuple[torch.Tensor, list[str]]:
    """Collate a batch into a single tensor and a list of env names.

    Returns:
        data:      (batch_size, N, 2, C, H, W)
        env_names: list of length batch_size
    """
    tensors, env_names = zip(*batch)
    return torch.stack(tensors), list(env_names)


class ActionGroupedDataset(Dataset):
    """Each item is N transitions from the same (env, action), shape (N, 2, C, H, W).

    All (env, action) pairs are enumerated and cycled round-robin so every
    pair contributes equally to the dataset regardless of per-env action
    space size.
    """

    def __init__(
        self,
        gym_envs: list[gym.Env],
        dataset_size: int,
        n_transitions: int = 10,
    ) -> None:
        super().__init__()
        self.n_transitions = n_transitions
        self.dataset: list[tuple[torch.Tensor, str]] = []

        env_action_pairs: list[tuple[gym.Env, int]] = [
            (env, action_idx)
            for env in gym_envs
            for action_idx in range(env.action_space.n)
        ]
        pair_cycle = itertools.cycle(env_action_pairs)

        while len(self.dataset) < dataset_size:
            env, action = next(pair_cycle)
            # collect_n_transitions raises RuntimeError if collection fails;
            # let it propagate so failures are never silently swallowed.
            item = collect_n_transitions(env, action, n_transitions)
            self.dataset.append((item, env.spec.id))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


# ---------------------------------------------------------------------------
# Deprecated dataset kept for reference only
# ---------------------------------------------------------------------------

class ChessDataset(Dataset):
    """Deprecated: use ActionGroupedDataset. Kept for reference."""

    def __init__(self, gym_envs: list[gym.Env], dataset_size: int) -> None:
        raise NotImplementedError(
            "ChessDataset is deprecated. Use ActionGroupedDataset instead."
        )


# ---------------------------------------------------------------------------
# New encoder
# ---------------------------------------------------------------------------

class MultiTransitionEncoder(nn.Module):
    """Encodes N same-action transitions into a single latent vector.

    Each transition is a (s_t, s_t+1) pair of observations (chess grids or
    RGB frames). The pair is concatenated along the channel dim, passed through
    a shared CNN to produce a token, then N tokens are processed by a
    Transformer encoder and mean-pooled into a single latent_dim vector.

    Input:  (batch_size, N, 2, C, H, W)
    Output: (batch_size, latent_dim)
    """

    def __init__(
        self,
        in_channels: int,        # C: 1 for chess grid, 3 for RGB robot  # OBS_CHANNELS_HOOK
        latent_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # CNN processes a concatenated (s_t, s_t+1) pair: 2C input channels.
        # AdaptiveMaxPool2d(4) makes the CNN resolution-independent, which is
        # important for sweeping over width/height.
        self.cnn = nn.Sequential(
            nn.Conv2d(2 * in_channels, 32, kernel_size=3, padding=1), # 32 is a hardcoded hyperparam
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 is also a hardcoded hyperparam
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(4),   # → (B*N, 64, 4, 4) # NOTE: not convinced there is a need for height/width invariance
            nn.Flatten(),              # → (B*N, 1024)
        )

        self.input_proj = nn.Linear(1024, d_model) # 1024 is determined by the aforementioned hardcoded hyperparams

        self.transformer = nn.TransformerEncoder( # NOTE: dim_feedforward might end up being too large
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.output_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, N, 2, C, H, W)

        Returns:
            (batch_size, latent_dim)
        """
        B, N, two, C, H, W = x.shape
        assert two == 2 and C == self.in_channels, (
            f"Expected x shape (B, N, 2, {self.in_channels}, H, W), got {x.shape}"
        )

        # Merge batch and N dims so the CNN processes all transitions at once
        x = x.contiguous().view(B * N, 2 * C, H, W)  # (B*N, 2C, H, W)
        x = self.cnn(x)                              # (B*N, 1024)
        x = self.input_proj(x)                       # (B*N, d_model)
        x = x.view(B, N, -1)                         # (B, N, d_model)
        x = self.transformer(x)                      # (B, N, d_model)
        x = x.mean(dim=1)                            # (B, d_model) # NOTE: not convinced mean is the right pool to use here
        return self.output_proj(x)                   # (B, latent_dim)


# ---------------------------------------------------------------------------
# New shared decoder
# ---------------------------------------------------------------------------

class SharedDecoder(nn.Module):
    """Shared decoder: reconstructs s_t+1 from an action embedding.

    Unlike the old per-environment Decoder, a single SharedDecoder handles
    all training environments. It receives target H and W at forward time,
    making it resolution-independent across sweep configurations.

    Input:  (batch_size, latent_dim)
    Output: (batch_size, out_channels, target_h, target_w)
    """

    def __init__(self, latent_dim: int, out_channels: int) -> None:  # OBS_CHANNELS_HOOK
        super().__init__()
        self.out_channels = out_channels

        self.fc = nn.Linear(latent_dim, 64 * 2 * 2) # 64 is hardcoded hyperparam. Not sure where * 2 *2 came from

        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2),                               # → (B, 64, 4, 4)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                               # → (B, 32, 8, 8) # NOTE: not convinced there is a need for height/width invariance
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),   # output normalised to [0, 1] — matches obs_to_tensor range
        )

    def forward(self, z: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Args:
            z:        (batch_size, latent_dim)
            target_h: desired output height
            target_w: desired output width

        Returns:
            (batch_size, out_channels, target_h, target_w)
        """
        x = self.fc(z)                              # (B, 64*2*2)
        x = x.view(x.shape[0], 64, 2, 2)           # (B, 64, 2, 2)
        x = self.deconv(x)                          # (B, out_channels, 8, 8)
        # Final resize to exact target dimensions
        x = nn.functional.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        ) # NOTE: don't love this, also used for height/width invariance
        return x


# ---------------------------------------------------------------------------
# Deprecated classes kept for checkpoint compatibility
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Deprecated: use MultiTransitionEncoder. Kept for checkpoint compatibility."""

    def __init__(self, height: int = config["height"], width: int = config["width"],
                 input_dim: int = config["height"] * config["width"], latent_dim: int = 3):
        super(Encoder, self).__init__()

        self.height = height
        self.width = width
        self.input_dim: int = input_dim # height by width
        self.latent_dim: int = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(2 * input_dim, 40), # 2*input_dim because input is a flattened _pair_ of observations
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input with shape (batch_size, 2, height, width) or (2, height, width). Output will have shape (batch_size, latent_dim)
        assert x.shape[x.dim()-3:x.dim()] == (2, self.height, self.width), \
            f"expected input shape (batch_size, 2, {self.height}, {self.width}), got {x.shape}"
        
        # insert explicit batch dimension if not already present
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size: int = x.shape[0]
        # Flatten input
        x = x.view(batch_size, -1)

        # Get latent representation
        latent: torch.Tensor = self.encoder(x)

        assert latent.shape == (batch_size, self.latent_dim), f"expected input with shape ({batch_size}, {self.latent_dim}), but got {latent.shape}"
        
        return latent


class Decoder(nn.Module):
    """Deprecated: use SharedDecoder. Kept for checkpoint compatibility."""

    def __init__(self, latent_dim: int = 3, # for chessworld, latent dim should be < 4 because transition can be exactly determined by 4 numbers
                 height: int = config["height"], width: int = config["width"],
                 output_dim: int = config["height"] * config["width"]):
        super(Decoder, self).__init__()

        self.height = height
        self.width = width
        self.latent_dim: int = latent_dim
        self.output_dim: int = output_dim # width by height

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, output_dim * 2), # 2*output_dim because output is a flattened _pair_ of observations
            nn.Tanh()
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Expect input with shape (batch_size, latent_dim). Output will have shape (batch_size, 2, height, width)
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
# each environment will share an encoder and a decoder
#
# For every env in training envs, train encoder-decoder on transition tuples.
# Tuples come from a random policy.
#
# Train stack by reconstruction loss. Encoder takes in several tuples and encodes
# it into a single embedding. Train the encoder-decoder stack by reconstruction (MSE loss) 
# against s_t+1 for each s_t+1 in the input tuples.

# loss function is MSE. CrossEntropyLoss would be better for chessworld 
# (since states are trinary - 0's where agent is not, 1 where agent is, 2 where 
# target is) but MSE generalizes to RGB images better.

# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main(primary_run_id: str, run_dir: Path = Path("models"), cfg: dict = config) -> None:
    """Train the MultiTransitionEncoder and SharedDecoder.

    Args:
        primary_run_id: W&B group ID. The IDM training run is placed in this
                        group alongside PPO worker runs.
        run_dir:        Directory to save model checkpoints. Created if it doesn't exist.
        cfg:            Full config dict. Defaults to the module-level idm_training_config_dict
                        for backward-compatible standalone usage
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])

    # Device configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels: int = cfg["obs_channels"]  # OBS_CHANNELS_HOOK

    training_envs: list[gym.Env] = [
        gym.make(env_name, width=cfg["width"], height=cfg["height"])
        for env_name in cfg["training_envs"]
    ]

    train_dataset: ActionGroupedDataset = ActionGroupedDataset(
        training_envs,
        dataset_size=cfg["dataset_size"],
        n_transitions=cfg["n_transitions_per_action"],
    )

    # each datum is a tensor with shape (2, width, height)
    # corresponding to a stack of s_t and s_t+1, where each is a 2d tensor with
    # shape (width, height)
    train_loader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    idm: MultiTransitionEncoder = MultiTransitionEncoder( # NOTE: currently, model is currently resolution agnostic, which I may want to change
        in_channels=in_channels,
        latent_dim=cfg["idm_latent_dim"],
        d_model=cfg["idm_d_model"],
        nhead=cfg["idm_nhead"],
        num_layers=cfg["idm_num_layers"],
    ).to(device)

    decoder: SharedDecoder = SharedDecoder( # NOTE: currently, model is currently resolution agnostic, which I may want to change
        latent_dim=cfg["idm_latent_dim"],
        out_channels=in_channels,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(idm.parameters()) + list(decoder.parameters()),
        lr=cfg["lr"],
    )

    # initialize wandb run
    with wandb.init(
        project=cfg["project_name"],
        entity=cfg["wandb_entity"],
        config=cfg,
        group=primary_run_id,
    ) as run:
        # --- training loop starts here ---
        num_epochs: int = cfg["epochs"]
        for epoch in range(num_epochs):
            total_loss: float = 0.0

            for batch_idx, (x, env_names) in enumerate(train_loader):
                # x: (B, N, 2, C, H, W)
                x = x.to(device)
                B, N, _, C, H, W = x.shape

                z: torch.Tensor = idm(x)  # (B, latent_dim)

                # Decode against each of the N s_t+1 targets independently.
                # Mean so loss scale is invariant to n_transitions_per_action
                per_n_losses: list[torch.Tensor] = []
                for i in range(N):
                    s_tp1_i = x[:, i, 1, :, :, :]      # (B, C, H, W) — s_t+1 only
                    recon_i = decoder(z, H, W)           # (B, C, H, W)
                    per_n_losses.append(criterion(recon_i, s_tp1_i))

                # aggregate losses
                batch_loss: torch.Tensor = torch.stack(per_n_losses).mean()

                # Backward pass and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

                # data logging and saving
                if batch_idx % 5 == 0:
                    # save models so far, for purposes of use with policy or 
                    # resuming training (versioned dict so _train_worker can
                    # detect and reject stale v1 MLP checkpoints)
                    torch.save(
                        {
                            "state_dict": idm.state_dict(),
                            "idm_version": 2,
                            "config": cfg,
                        },
                        run_dir / "idm.pt",
                    )
                    torch.save(decoder.state_dict(), run_dir / "shared_decoder.pt")
                    
                    # save optimizer state for resuming training
                    torch.save(optimizer.state_dict(), run_dir / "optimizer.pt")

                    # record training stats
                    run.log({
                        "batch_loss": batch_loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                    })

                    # Per-env loss for diagnostics (no gradient needed)
                    with torch.no_grad():
                        for env_name in set(env_names):
                            mask = [i for i, n in enumerate(env_names) if n == env_name]
                            env_loss = criterion(
                                decoder(z[mask], H, W),
                                x[mask, 0, 1],   # s_t+1 of first transition as proxy
                            )
                            short_name: str = env_name[str.rfind(env_name, "/") + 1:]
                            run.log({
                                f"{short_name}_loss": env_loss.item(),
                                "epoch": epoch,
                                "batch": batch_idx,
                            })

            avg_loss: float = total_loss / len(train_loader)
            run.log({"epoch_loss_averaged_over_batches": avg_loss, "epoch": epoch})
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main(primary_run_id="standalone")

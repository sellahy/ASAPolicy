# Action Space Agnostic Reinforcement Learning

This project investigates whether a single reinforcement learning policy can operate across environments with fundamentally different action spaces — without any retraining or architectural changes per environment. The approach uses an **Inverse Dynamics Model (IDM)** to map state transitions into a shared latent action space, and a **Transformer-based ASA policy** that reasons over latent action embeddings rather than raw action indices.

The ASA policy is compared against per-environment baseline policies on three metrics: **jumpstart** (initial performance in a new environment), **learning speed** (how fast each policy converges), and **asymptotic performance** (final return after training). All metrics are tracked via Weights & Biases.

---

## Repository Structure

```
.
├── chess_env/                  # Custom Gymnasium environments (installable package)
│   ├── envs/                   # Environment implementations
│   │   ├── chessworld.py       # Base environment (toroidal grid, Chebyshev reward)
│   │   ├── bishopworld.py      # Bishop movement (diagonal)
│   │   ├── camelworld.py       # Camel movement (3,1 leaper)
│   │   ├── goldgeneralworld.py # Gold General movement (shogi)
│   │   ├── kingworld.py        # King movement (all 8 adjacents)
│   │   ├── knightworld.py      # Knight movement (L-shapes)
│   │   ├── silvergeneralworld.py # Silver General movement (shogi)
│   │   └── zebraworld.py       # Zebra movement (3,2 leaper)
│   ├── wrappers/               # Gymnasium wrappers (ClipReward, RelativePosition, etc.)
│   └── pyproject.toml          # Package definition for chess_env
│
├── config.py                   # All hyperparameters (IDM + PPO + plotting)
├── idm_training.py             # IDM encoder/decoder training
├── ppo_clip.py                 # PPO-Clip training, ASAPolicy, network definitions
├── handle_data.py              # W&B data retrieval utilities
├── plot_results.py             # Generates comparison figures from W&B data
├── utils.py                    # Shared math utilities (smoothing, convergence helpers)
├── run_experiment.py           # Single entry point for the full pipeline
├── sweep_config.yaml           # W&B hyperparameter sweep configuration
└── requirements.txt            # Python dependencies
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sellahy/ASAPolicy.git

# Install dependencies (includes the chess_env package in editable mode)
pip install -r requirements.txt

# Log in to Weights & Biases (required for all training and plotting)
wandb login
```

Before running, set your W&B entity in `config.py`:

```python
# config.py
idm_training_config_dict = {
    "wandb_entity": "your-wandb-username-or-team",
    ...
}
```

---

## Configuration

All hyperparameters live in `config.py`. The file defines three dicts:

| Dict | Purpose |
|---|---|
| `idm_training_config_dict` | IDM encoder training (grid size, learning rate, environments, latent dim, etc.) |
| `ppo_config_dict` | PPO training (timesteps, gamma, clip epsilon, ASA architecture, plotting thresholds) |
| `base_config` | Merged dict of both — this is what `run_experiment.py` and W&B sweeps consume |

Key parameters to know before running:

```python
# Which environments to train the IDM on
"training_envs": ["chess_env/ChessWorld-v0", "chess_env/BishopWorld-v0", ...]

# Which environments the ASA agent trains on (must be a subset of training_envs)
"asa_envs": ["chess_env/ChessWorld-v0", "chess_env/BishopWorld-v0", ...]

# Grid size (all environments share the same size)
"width": 5, "height": 5

# Total PPO training budget per agent
"total_timesteps": 300_000

# Number of evenly-spaced intermediate checkpoints saved during PPO training.
# Checkpoint timesteps = [k * total_timesteps / num_checkpoints for k in 1..num_checkpoints].
"num_checkpoints": 4

# Convergence criterion for computing T_transfer (the transfer fine-tuning budget).
# A metric is converged at the first point (i.e. after the last violation) where
# the relative change between consecutive smoothed datapoints drops below this fraction.
# Applied to all four logged metrics (return, ep_len, policy_loss, value_loss).
"transfer_relative_change_threshold": 0.15
```

---

## Running the Pipeline

The full pipeline — IDM training, PPO training, transfer evaluation, and visualization — is controlled by a single script:

```bash
python run_experiment.py
```

**Stage auto-detection**: completed stages are detected by the presence of their output checkpoints. Re-running the same command after an interruption automatically resumes from where it left off, reusing the same W&B run.

### Flags

| Flag | Effect |
|---|---|
| *(no flags)* | Run all stages; auto-skip any already completed |
| `--skip-plot` | Run training stages but skip plot generation at the end |
| `--force` | Ignore existing checkpoints and re-run all stages from scratch |
| `--save_dir <path>` | Directory to save plot images (default: `figures/`) |

### Pipeline stages

1. **IDM training** — trains a shared encoder and one decoder per environment on random-policy transition pairs. Saves `idm.pt` and `<env>_decoder.pt` to the run directory.
2. **PPO training** — trains one baseline agent per environment and one ASA agent across `asa_envs`, in parallel across available GPUs. Saves final `<agent>_policy.pt` checkpoints plus `num_checkpoints` intermediate `<agent>_policy_step{N}.pt` checkpoints per agent.
3. **Checkpoint transfer** — computes `T_transfer` (the minimum convergence timestep across all training agents, measured as the first stable point where the relative change between consecutive smoothed metric values drops below `transfer_relative_change_threshold` across all four logged metrics). Then, for every saved checkpoint of every training agent, fine-tunes that checkpoint on each unseen environment for exactly `T_transfer` timesteps. `T_transfer` is persisted to `t_transfer.json` so the stage can resume without re-querying W&B. The result is a transfer performance curve indexed by pretraining depth.
4. **Visualization** — generates comparison plots from W&B history and saves them locally.

### Checkpoint and run directory layout

Each run's files are stored under `runs/<wandb_run_id>/`:

```
runs/
  <wandb_run_id>/
    config.json                           # Full resolved config for this run
    wandb_run_id.txt                      # Stored W&B run ID for resumption
    t_transfer.json                       # Computed T_transfer budget for stage 3
    idm.pt                                # Trained IDM encoder
    <env>_decoder.pt                      # Per-environment IDM decoders
    <env>_baseline_policy.pt              # Final baseline policy checkpoints
    <env>_baseline_policy_step{N}.pt      # Intermediate baseline checkpoints
    <env>_baseline_value_step{N}.pt       # Intermediate baseline value nets
    asa_agent_policy.pt                   # Final trained ASA policy
    asa_agent_policy_step{N}.pt           # Intermediate ASA checkpoints
    asa_agent_value_step{N}.pt            # Intermediate ASA value nets
    asa_ckpt{N}_transfer_<env>_policy.pt  # Checkpoint-transfer ASA results
    <src>_baseline_ckpt{N}_transfer_<env>_policy.pt  # Checkpoint-transfer baseline results
```

---

## Visualizing Results

Plots can be generated for any completed run at any time, independently of training:

```bash
python plot_results.py --run_id <wandb_run_id> --save_dir figures/
```

This downloads the run's metric history from W&B and generates seven figures:

| Figure | What it shows |
|---|---|
| `learning_curves.png` | Smoothed return vs. step, one subplot per environment, all agents overlaid |
| `jumpstart.png` | Mean return over the first 100 eval events per agent per environment |
| `asymptotic.png` | Mean return over the last 100 eval events per agent per environment |
| `return_convergence_speed.png` | First step where return reaches 80% of asymptote (lower = faster) |
| `policy_loss_convergence_speed.png` | First step where policy loss drops to 80% of its initial value |
| `value_loss_convergence_speed.png` | First step where value loss drops to 80% of its initial value |
| `transfer_vs_checkpoint_depth.png` | Mean transfer return vs. pretraining checkpoint timestep per unseen environment — reveals whether more pretraining helps or hurts transfer |

For hyperparameter sensitivity plots across a sweep:

```bash
python plot_results.py --run_id <any_run_id> \
    --hyperparams gamma ppo_lr latent_action_dim dim_feedforward \
    --save_dir figures/
```

---

## Hyperparameter Sweeps

Create sweep with following bash command:
wandb sweep --project action_space_agnostic_agent sweep_config.yaml
it will print a sweep_id to use in the following command. Set for resuming with the following CLI command:
wandb sweep entity/project/sweep_id --resume

---

## Data Utilities

`handle_data.py` provides functions for inspecting saved run data from Python:

```python
from handle_data import get_runs, get_config, get_all_envs, load_run_as_dataframe

# List all run IDs in the project
run_ids = get_runs()

# Get the config for a specific run
cfg = get_config("abc123")

# Get all environments used in a run
envs = get_all_envs("abc123")

# Load all metrics as a tidy DataFrame
df = load_run_as_dataframe("abc123")
# Columns: run_id, agent_name, env, metric, value, step, training_timestep
# training_timestep is the actual PPO training timestep at which the eval was
# logged (read from the same wandb.log() call as the metrics). NaN on rows
# that originate from IDM training.
```

---

## Environments

All environments share the same base class (`ChessWorldEnv`): a 5×5 toroidal grid where the agent navigates to a fixed target. The reward at each step is the negative Chebyshev distance to the target (accounting for wrap-around).

| Environment | Piece | Actions | Movement |
|---|---|---|---|
| `ChessWorld-v0` | Rook-like | 5 | ±1 cardinal + no-op |
| `BishopWorld-v0` | Bishop | 5 | ±1 diagonal + no-op |
| `KingWorld-v0` | King | 9 | All 8 adjacent + no-op |
| `KnightWorld-v0` | Knight | 9 | L-shapes (±2,±1) + no-op |
| `GoldGeneralWorld-v0` | Gold General | 7 | Cardinal + up-diagonals + no-op |
| `SilverGeneralWorld-v0` | Silver General | 6 | Up + all diagonals + no-op |
| `CamelWorld-v0` | Camel | 9 | (3,1) leaps + no-op |
| `ZebraWorld-v0` | Zebra | 9 | (3,2) leaps + no-op |

`DragonHorseWorld` and `DragonKingWorld` exist in `chess_env/envs/` but are disabled pending a fix to their movement logic.

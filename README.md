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
├── run_experiment.py           # Single entry point for the full pipeline
├── sweep_config.yaml           # W&B hyperparameter sweep configuration
├── requirements.txt            # Python dependencies
└── research.md                 # Technical notes and design rationale
```

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd ClassProject

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
"total_timesteps": 200_000
```

---

## Running the Pipeline

The full pipeline — IDM training, PPO training, transfer evaluation, and visualization — is controlled by a single script:

```bash
python run_experiment.py
```

**Stage auto-detection**: completed stages are detected by the presence of their output checkpoints. Re-running the same command after an interruption automatically resumes from where it left off, reusing the same W&B run.

Create sweep with following bash command:
wandb sweep --project action_space_agnostic_agent sweep_config.yaml
it will print a sweep_id to use in the following command. set for resuming with:
wandb sweep entity/project/sweep_id --resume
for a SLURM cluster, insert the sweep_id into launch_sweep.sh and run the following command to start the runs:
sbatch launch_sweep.sh  
If it gets interrupted, it can simply be rerun and it will continue where it left off.

### Flags

| Flag | Effect |
|---|---|
| *(no flags)* | Run all stages; auto-skip any already completed |
| `--skip-plot` | Run training stages but skip plot generation at the end |
| `--force` | Ignore existing checkpoints and re-run all stages from scratch |
| `--save_dir <path>` | Directory to save plot images (default: `figures/`) |

### Pipeline stages

1. **IDM training** — trains a shared encoder and one decoder per environment on random-policy transition pairs. Saves `idm.pt` and `<env>_decoder.pt` to the run directory.
2. **PPO training** — trains one baseline agent per environment and one ASA agent across `asa_envs`, in parallel across available GPUs. Saves `<agent>_policy.pt` checkpoints.
3. **Transfer training** — for each environment the ASA agent did not train in, loads the saved ASA checkpoint fresh and trains on that environment independently. Measures jumpstart and transfer learning speed.
4. **Data storage** — implicit; all metrics are logged live to W&B throughout stages 1–3.
5. **Visualization** — generates comparison plots from W&B history and saves them locally.

### Checkpoint and run directory layout

Each run's files are stored under `runs/<wandb_run_id>/`:

```
runs/
  <wandb_run_id>/
    config.json               # Full resolved config for this run
    wandb_run_id.txt          # Stored W&B run ID for resumption
    idm.pt                    # Trained IDM encoder
    <env>_decoder.pt          # Per-environment IDM decoders
    <env>_baseline_policy.pt  # Baseline policy checkpoints
    asa_agent_policy.pt       # Trained ASA policy
    asa_transfer_<env>_policy.pt  # Transfer-trained ASA checkpoints
```

---

## Visualizing Results

Plots can be generated for any completed run at any time, independently of training:

```bash
python plot_results.py --run_id <wandb_run_id> --save_dir figures/
```

This downloads the run's metric history from W&B and generates six figures:

| Figure | What it shows |
|---|---|
| `learning_curves.png` | Smoothed return vs. step, one subplot per environment, all agents overlaid |
| `jumpstart.png` | Mean return over the first 100 eval events per agent per environment |
| `asymptotic.png` | Mean return over the last 100 eval events per agent per environment |
| `return_convergence_speed.png` | First step where return reaches 80% of asymptote (lower = faster) |
| `policy_loss_convergence_speed.png` | First step where policy loss drops to 80% of its initial value |
| `value_loss_convergence_speed.png` | First step where value loss drops to 80% of its initial value |

For hyperparameter sensitivity plots across a sweep:

```bash
python plot_results.py --run_id <any_run_id> \
    --hyperparams gamma ppo_lr latent_action_dim dim_feedforward \
    --save_dir figures/
```

---

## Hyperparameter Sweeps

Sweeps use W&B's Bayesian optimization to search over the parameter space defined in `sweep_config.yaml`.

```bash
# Register the sweep with W&B (prints a sweep ID)
wandb sweep sweep_config.yaml

# Launch agents — NUM_AGENTS controls parallelism independently of GPU count.
# Agents are assigned GPUs round-robin, so you can run more agents than GPUs.
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l || echo 1)
NUM_AGENTS=${NUM_AGENTS:-$NUM_GPUS}   # default: one agent per GPU; override freely
for i in $(seq 0 $((NUM_AGENTS - 1))); do
  GPU_ID=$((i % NUM_GPUS))
  CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent <entity/project/sweep_id> &
done
wait
```

To run 4 agents on 2 GPUs:

```bash
NUM_AGENTS=4 bash -c '
  NUM_GPUS=$(nvidia-smi --list-gpus | wc -l || echo 1)
  for i in $(seq 0 $((NUM_AGENTS - 1))); do
    CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) wandb agent <sweep_id> &
  done
  wait
'
```

Each W&B agent repeatedly pulls a new hyperparameter configuration from the sweep server, runs the full `run_experiment.py` pipeline for that configuration, and reports results back. W&B's Bayesian optimizer uses completed results to suggest better configurations.

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
# Columns: run_id, agent_name, env, metric, value, step
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

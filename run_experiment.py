"""run_experiment.py

Single entry point for the full IDM + PPO training and evaluation pipeline.

Stages:
    1. IDM training       — trains shared encoder + per-env decoders
    2. PPO training       — trains all baseline agents and the ASA agent in parallel
    3. ASA transfer       — fine-tunes ASA on each unseen environment independently
    4. W&B data storage   — implicit; metrics are logged live throughout stages 1-3
    5. Visualization      — generates comparison plots from W&B run history

Completed stages are auto-detected by the presence of their output checkpoints,
so an interrupted run can be resumed by simply re-running this script with the
same config.

Usage:
    python run_experiment.py                   # full pipeline, or auto-resume
    python run_experiment.py --skip-plot       # no plots after training
    python run_experiment.py --force           # re-run all stages from scratch
    python plot_results.py --run_id <id>       # visualize any completed run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from config import base_config
from idm_training import main as train_idm
from ppo_clip import run_all_agents, run_asa_transfer


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Config hash (used to derive a stable run directory from the config)
# ---------------------------------------------------------------------------

def config_hash(cfg: dict) -> str:
    """Return a short deterministic hash of the config dict.

    Keys are sorted before hashing so insertion order does not affect the
    result. Used to derive a stable run directory path: the same config always
    maps to the same directory, and a different config gets a fresh one.

    Args:
        cfg: Config dictionary to hash.

    Returns:
        12-character hex string.
    """
    serialized: str = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Stage completion detection
# ---------------------------------------------------------------------------

def _expected_policy_files(cfg: dict) -> list[str]:
    """Return the checkpoint filenames that stage 2 (PPO training) produces.

    Args:
        cfg: Experiment config dict.

    Returns:
        List of filename strings (not full paths).
    """
    baseline_names: list[str] = [
        name.replace("chess_env/", "").replace("-v0", "") + "_baseline_policy.pt"
        for name in cfg["training_envs"]
    ]
    return baseline_names + ["asa_agent_policy.pt"]


def _expected_transfer_files(cfg: dict) -> list[str]:
    """Return the checkpoint filenames that stage 3 (transfer) produces.

    Args:
        cfg: Experiment config dict.

    Returns:
        List of filename strings (not full paths).
    """
    unseen_envs: list[str] = [
        name for name in cfg["training_envs"]
        if name not in cfg["asa_envs"]
    ]
    return [
        f"asa_transfer_{name.replace('chess_env/', '').replace('-v0', '')}_policy.pt"
        for name in unseen_envs
    ]


def _stage1_done(run_dir: Path) -> bool:
    """Return True if IDM training output (idm.pt) already exists."""
    return (run_dir / "idm.pt").exists()


def _stage2_done(run_dir: Path, cfg: dict) -> bool:
    """Return True if all PPO policy checkpoints already exist."""
    return all((run_dir / f).exists() for f in _expected_policy_files(cfg))


def _stage3_done(run_dir: Path, cfg: dict) -> bool:
    """Return True if all transfer policy checkpoints already exist."""
    return all((run_dir / f).exists() for f in _expected_transfer_files(cfg))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full IDM + PPO training and evaluation pipeline. "
            "Completed stages are detected automatically and skipped — "
            "re-running this script with the same config resumes from where "
            "it left off."
        )
    )
    parser.add_argument(
        "--skip-plot", action="store_true",
        help="Skip plot generation after training completes."
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Ignore existing checkpoints and re-run all stages from scratch, "
            "overwriting prior results."
        )
    )
    parser.add_argument(
        "--save_dir", type=str, default="figures",
        help="Directory to save plots (used when --skip-plot is not set)."
    )
    args = parser.parse_args()

    # Derive the run directory from the config hash before W&B init so that
    # we can look up an existing W&B run ID for this config.
    run_dir: Path = Path("runs") / config_hash(base_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_id_file: Path = run_dir / "wandb_run_id.txt"

    if args.force:
        # Remove all checkpoints so every stage re-runs
        for f in run_dir.glob("*.pt"):
            f.unlink()
        if wandb_id_file.exists():
            wandb_id_file.unlink()

    # Resume the existing W&B run for this config if one was started before,
    # so all metrics land on the same run page regardless of interruptions.
    if wandb_id_file.exists():
        existing_run_id: str = wandb_id_file.read_text().strip()
        run = wandb.init(
            project=base_config.get("project_name", "chess_asa"),
            entity=base_config.get("wandb_entity"),
            id=existing_run_id,
            resume="must",
            config=base_config,
        )
    else:
        run = wandb.init(
            project=base_config.get("project_name", "chess_asa"),
            entity=base_config.get("wandb_entity"),
            config=base_config,
        )
        wandb_id_file.write_text(run.id)

    config: dict = dict(run.config)
    set_seeds(config["seed"])

    # Persist the resolved config alongside checkpoints for local reference
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Stage 1: IDM training
    # ------------------------------------------------------------------
    if _stage1_done(run_dir) and not args.force:
        print(f"=== Stage 1: Skipped — idm.pt already exists in {run_dir} ===")
    else:
        print("=== Stage 1: IDM training ===")
        train_idm(config, run_dir)

    # ------------------------------------------------------------------
    # Stage 2: PPO training of all agents
    # ------------------------------------------------------------------
    if _stage2_done(run_dir, config) and not args.force:
        print("=== Stage 2: Skipped — all policy checkpoints already exist ===")
    else:
        print("=== Stage 2: PPO training ===")
        # Set spawn start method for CUDA compatibility before launching workers
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # start method already set
        run_all_agents(config, str(run_dir))

    # ------------------------------------------------------------------
    # Stage 3: ASA transfer training on unseen environments
    # ------------------------------------------------------------------
    if _stage3_done(run_dir, config) and not args.force:
        print("=== Stage 3: Skipped — all transfer checkpoints already exist ===")
    else:
        print("=== Stage 3: ASA transfer training ===")
        run_asa_transfer(config, str(run_dir))

    # ------------------------------------------------------------------
    # Stage 4: W&B data (implicit — metrics logged live in stages 1-3)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stage 5: Visualization
    # ------------------------------------------------------------------
    if not args.skip_plot:
        print("=== Stage 5: Generating plots ===")
        from plot_results import (
            plot_asymptotic,
            plot_jumpstart,
            plot_learning_curves,
            plot_loss_convergence_speed,
            plot_return_convergence_speed,
        )
        from handle_data import get_all_envs, get_config, load_run_as_dataframe

        os.makedirs(args.save_dir, exist_ok=True)

        df: object       = load_run_as_dataframe(run.id)
        envs: list[str]  = get_all_envs(run.id)
        cfg: dict        = get_config(run.id)
        K: int           = int(cfg.get("jumpstart_K", 100))
        K_late: int      = int(cfg.get("asymptote_K", 100))
        thresh: float    = float(cfg.get("convergence_threshold", 0.8))

        plot_learning_curves(df, envs, save_dir=args.save_dir)
        plot_jumpstart(df, K=K, save_dir=args.save_dir)
        plot_asymptotic(df, K=K_late, save_dir=args.save_dir)
        plot_return_convergence_speed(df, threshold=thresh, save_dir=args.save_dir)
        plot_loss_convergence_speed(df, threshold=thresh, save_dir=args.save_dir)
        print(f"Plots saved to {args.save_dir}/")
    else:
        print("=== Stage 5: Skipped ===")

    run.finish()
    print(f"Done. Run ID: {run.id}  |  Run dir: {run_dir}")


if __name__ == "__main__":
    main()

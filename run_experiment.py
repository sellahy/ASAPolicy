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
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

from config import base_config
from idm_training import main as train_idm
from ppo_clip import run_all_agents, run_asa_transfer, run_baseline_transfer, run_checkpoint_transfer

# SIGTERM handling (intended for enabling resuming sweeps on SLURM cluster)
def handler(signum, frame):
    wandb.run.mark_preempting()
    sys.exit(128 + signum) 
signal.signal(signal.SIGTERM, handler)


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
    """Return True if all final and intermediate policy checkpoints exist."""
    # Final checkpoints (existing check)
    if not all((run_dir / f).exists() for f in _expected_policy_files(cfg)):
        return False

    # Intermediate checkpoints for each training agent
    num_ckpts: int = cfg.get("num_checkpoints", 4)
    total_ts: int  = cfg["total_timesteps"]
    ckpt_steps: list[int] = sorted(set(
        int(round(k * total_ts / num_ckpts))
        for k in range(1, num_ckpts + 1)
    ))

    all_agents: list[str] = [
        name.replace("chess_env/", "").replace("-v0", "") + "_baseline"
        for name in cfg["training_envs"]
    ] + ["asa_agent"]

    for agent in all_agents:
        for step in ckpt_steps[:-1]:  # skip last — covered by the final checkpoint
            if not (run_dir / f"{agent}_policy_step{step}.pt").exists():
                return False
    return True


def _checkpoint_transfer_agent_names(cfg: dict) -> list[str]:
    """Return the agent names that run_checkpoint_transfer() will produce.

    Mirrors the enumeration logic in run_checkpoint_transfer() so that
    _stage3_checkpoint_done() can check completion without spawning processes.
    """
    import gymnasium as gym
    import chess_env  # noqa: F401 — registers envs

    num_ckpts: int = cfg.get("num_checkpoints", 4)
    total_ts: int  = cfg["total_timesteps"]
    ckpt_steps: list[int] = sorted(set(
        int(round(k * total_ts / num_ckpts))
        for k in range(1, num_ckpts + 1)
    ))

    unseen_envs: list[str] = [
        name for name in cfg["all_envs"]
        if name not in cfg["asa_envs"]
    ]

    # Action-space size map for baseline matching
    size_to_source: dict[int, list[str]] = {}
    for name in cfg["asa_envs"]:
        env = gym.make(name, width=cfg["width"], height=cfg["height"])
        size_to_source.setdefault(env.action_space.n, []).append(name)
        env.close()

    agent_names: list[str] = []

    for ckpt_step in ckpt_steps:
        for target_name in unseen_envs:
            target_tag: str = target_name.replace("chess_env/", "").replace("-v0", "")
            agent_names.append(f"asa_ckpt{ckpt_step}_transfer_{target_tag}")

        for source_name in cfg["asa_envs"]:
            source_tag: str = source_name.replace("chess_env/", "").replace("-v0", "")
            env = gym.make(source_name, width=cfg["width"], height=cfg["height"])
            source_n: int = env.action_space.n
            env.close()
            for target_name in unseen_envs:
                env = gym.make(target_name, width=cfg["width"], height=cfg["height"])
                target_n: int = env.action_space.n
                env.close()
                if target_n != source_n:
                    continue
                target_tag = target_name.replace("chess_env/", "").replace("-v0", "")
                agent_names.append(
                    f"{source_tag}_baseline_ckpt{ckpt_step}_transfer_{target_tag}"
                )

    return agent_names


def _stage3_checkpoint_done(run_dir: Path, cfg: dict) -> bool:
    """Return True if all checkpoint-transfer policy files exist."""
    for agent_name in _checkpoint_transfer_agent_names(cfg):
        if not (run_dir / f"{agent_name}_policy.pt").exists():
            return False
    return True


def _stage3_done(run_dir: Path, cfg: dict) -> bool:
    """Return True if all transfer policy checkpoints already exist."""
    return all((run_dir / f).exists() for f in _expected_transfer_files(cfg))


def _expected_baseline_transfer_files(cfg: dict) -> list[str]:
    """Return checkpoint filenames that stage 3b (baseline transfer) produces.

    Replicates the matching logic from run_baseline_transfer so stage completion
    can be checked without spawning any processes.

    Args:
        cfg: Experiment config dict.

    Returns:
        List of filename strings (not full paths). Empty list if no baselines
        match any unseen environment.
    """
    import gymnasium as gym
    import chess_env  # registers envs with gymnasium

    size_to_source: dict[int, list[str]] = {}
    for name in cfg["asa_envs"]:
        env = gym.make(name, width=cfg["width"], height=cfg["height"])
        size_to_source.setdefault(env.action_space.n, []).append(name)
        env.close()

    files: list[str] = []
    unseen: list[str] = [n for n in cfg["all_envs"] if n not in cfg["asa_envs"]]
    for target_name in unseen:
        env = gym.make(target_name, width=cfg["width"], height=cfg["height"])
        target_n: int = env.action_space.n
        env.close()
        for source_name in size_to_source.get(target_n, []):
            source_tag: str = source_name.replace("chess_env/", "").replace("-v0", "")
            target_tag: str = target_name.replace("chess_env/", "").replace("-v0", "")
            agent_name: str = f"baseline_transfer_{source_tag}_to_{target_tag}"
            files += [f"{agent_name}_policy.pt", f"{agent_name}_value.pt"]
    return files


def _stage3b_done(run_dir: Path, cfg: dict) -> bool:
    """Return True if all baseline transfer checkpoints already exist.

    Returns True immediately (treating the stage as vacuously complete) if no
    training baseline has an action space matching any unseen environment.
    """
    files: list[str] = _expected_baseline_transfer_files(cfg)
    if not files:
        return True
    return all((run_dir / f).exists() for f in files)

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
    parser.add_argument(
        "--sweep", action="store_true", 
        help=(
            "indicates that this invocation of this file is part of a sweep "
            "and overrides config hyperparams using sweep_config.yaml"
        )
    )
    args = parser.parse_args()

    if args.sweep: # NOTE: in this block, force arg is ignored
        run = wandb.init(
            project=base_config.get("project_name"),
            entity=base_config.get("wandb_entity"),
            config=base_config,
        )
        # sweep overrides are now applied
        
        run_dir: Path = Path("runs") / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
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
                project=base_config.get("project_name"),
                entity=base_config.get("wandb_entity"),
                id=existing_run_id,
                resume="must",
                config=base_config,
            )
        else:
            run = wandb.init(
                project=base_config.get("project_name"),
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
        train_idm(run.id, run_dir, config)

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
        run_all_agents(run.id, config, str(run_dir))

    # ------------------------------------------------------------------
    # Stage 3: Checkpoint-based transfer evaluation
    # Compute T_transfer (minimum convergence timestep across all training
    # agents) then fine-tune every (agent × checkpoint) pair on each unseen
    # environment for exactly T_transfer timesteps.
    # ------------------------------------------------------------------

    # Compute and persist T_transfer so stage 3 can resume without re-querying
    # W&B if interrupted.
    t_transfer_file: Path = run_dir / "t_transfer.json"
    if t_transfer_file.exists() and not args.force:
        t_transfer: int = json.loads(t_transfer_file.read_text())["t_transfer"]
        print(f"=== T_transfer loaded from file: {t_transfer} ===")
    else:
        from handle_data import get_min_convergence_step, load_group_as_dataframe
        training_agent_names: list[str] = [
            name.replace("chess_env/", "").replace("-v0", "") + "_baseline"
            for name in config["training_envs"]
        ] + ["asa_agent"]
        training_env_names: list[str] = [
            name.replace("chess_env/", "")
            for name in config["training_envs"]
        ]
        # t_transfer = get_min_convergence_step(
        #     run.id, # FIXME: this results in "KeyError: 'agent_name'" in handle_data.compute_convergence_step because the run_id being passed here is the run that recorded the IDM stats, not a run that recorded a baseline or ASAPolicy training
        #     training_agent_names,
        #     training_env_names,
        #     total_timesteps=config["total_timesteps"],
        #     threshold=config.get("transfer_relative_change_threshold", 0.25),
        # )
        group_df = load_group_as_dataframe(run.id)          # all worker runs share group=run.id
        t_transfer = get_min_convergence_step(
            run_id=None,                                     # not used when df is passed directly
            training_agent_names=training_agent_names,
            training_env_names=training_env_names,
            total_timesteps=config["total_timesteps"],
            threshold=config.get("transfer_relative_change_threshold", 0.25),
            _df=group_df,
)

        t_transfer_file.write_text(json.dumps({"t_transfer": t_transfer}))
        print(f"=== T_transfer computed: {t_transfer} ===")

    if _stage3_checkpoint_done(run_dir, config) and not args.force:
        print("=== Stage 3: Skipped — all checkpoint transfer files exist ===")
    else:
        print(f"=== Stage 3: Checkpoint transfer (T_transfer={t_transfer}) ===")
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        run_checkpoint_transfer(run.id, config, str(run_dir), t_transfer)

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
            plot_transfer_vs_checkpoint_depth,
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

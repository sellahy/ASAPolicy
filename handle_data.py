"""handle_data.py

Utilities for retrieving and structuring experiment data from W&B.
All metric data is stored in W&B; no local pickle files are needed.

Usage:
    from handle_data import get_runs, get_config, load_run_as_dataframe

    for run_id in get_runs():
        df = load_run_as_dataframe(run_id)
        ...
"""

import wandb
import pandas as pd
from config import base_config

# W&B API client — reads credentials from the environment / ~/.netrc
_api: wandb.Api = wandb.Api()

# Project path in W&B: "entity/project_name"
PROJECT: str = f"{base_config['wandb_entity']}/{base_config['project_name']}"


def get_runs() -> list[str]:
    """Return a list of all available run IDs in the project.

    Returns:
        List of W&B run ID strings.
    """
    return [run.id for run in _api.runs(PROJECT)]


def get_config(run_id: str) -> dict:
    """Return the config dict logged for the given run.

    Args:
        run_id: W&B run ID string.

    Returns:
        Config dictionary as stored in W&B.
    """
    return dict(_api.run(f"{PROJECT}/{run_id}").config)


def get_all_envs(run_id: str) -> list[str]:
    """Return all environment names used in the given run.

    Reads training_envs from the run's stored config, which covers all
    environments used for IDM training, baseline training, ASA training,
    and transfer evaluation.

    Args:
        run_id: W&B run ID string.

    Returns:
        List of environment name strings (e.g. "chess_env/KingWorld-v0").
    """
    cfg: dict = get_config(run_id)
    return list(cfg.get("training_envs", []))


def get_env_for_agent(run_id: str, agent_name: str) -> list[str]:
    """Return the environments an agent logged metrics for in a given run.

    Metrics are logged as "{agent_name}/{env_name}/{metric}", so scanning
    the history column names is sufficient to infer the environments.

    Args:
        run_id:     W&B run ID string.
        agent_name: Agent name prefix as used in wandb.log() calls.

    Returns:
        List of environment name strings the agent trained in.
    """
    run = _api.run(f"{PROJECT}/{run_id}")
    keys: list[str] = run.history(samples=1).columns.tolist()
    prefix: str = f"{agent_name}/"
    envs: set[str] = set()
    for key in keys:
        if key.startswith(prefix):
            parts: list[str] = key.split("/")
            if len(parts) >= 3:  # agent_name / env_name / metric
                envs.add(parts[1])
    return list(envs)


def get_agent_for_env(run_id: str, env_name: str) -> list[str]:
    """Return all agent names that logged metrics for a given environment.

    Args:
        run_id:   W&B run ID string.
        env_name: Environment name as used in metric keys (e.g. "KingWorld-v0").

    Returns:
        List of agent name strings that trained in that environment.
    """
    run = _api.run(f"{PROJECT}/{run_id}")
    keys: list[str] = run.history(samples=1).columns.tolist()
    agents: set[str] = set()
    for key in keys:
        parts: list[str] = key.split("/")
        if len(parts) >= 3 and parts[1] == env_name:
            agents.add(parts[0])
    return list(agents)


def load_run_as_dataframe(run_id: str,
                          max_samples: int = 10_000) -> pd.DataFrame:
    """Download metric history for a run and return it as a tidy DataFrame.

    Parses all metric keys of the form ``{agent_name}/{env_name}/{metric}``
    where metric is one of: return, ep_len, policy_loss, value_loss.

    Args:
        run_id:      W&B run ID string.
        max_samples: Maximum number of history rows to download. W&B
                     sub-samples history for large runs; increase this
                     if curves appear coarse.

    Returns:
        DataFrame with columns:
            run_id, agent_name, env, metric, value, step
    """
    run = _api.run(f"{PROJECT}/{run_id}")
    history: pd.DataFrame = run.history(samples=max_samples)

    rows: list[dict] = []
    for col in history.columns:
        parts: list[str] = col.split("/")
        # Only process keys in the form: agent_name/env_name/metric_name
        if len(parts) != 3:
            continue
        agent_name, env_name, metric_name = parts
        if metric_name not in ("return", "ep_len", "policy_loss", "value_loss"):
            continue
        for _, row in history[["_step", col]].dropna().iterrows():
            rows.append({
                "run_id":     run_id,
                "agent_name": agent_name,
                "env":        env_name,
                "metric":     metric_name,
                "value":      float(row[col]),
                "step":       int(row["_step"]),
            })

    return pd.DataFrame(rows)


def load_all_runs_as_dataframe(max_samples: int = 10_000) -> pd.DataFrame:
    """Concatenate metric history from all runs in the project.

    Args:
        max_samples: Passed through to load_run_as_dataframe for each run.

    Returns:
        Concatenated DataFrame across all runs, with the same columns as
        load_run_as_dataframe.
    """
    frames: list[pd.DataFrame] = [
        load_run_as_dataframe(run_id, max_samples)
        for run_id in get_runs()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

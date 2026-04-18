"""handle_data.py

Utilities for retrieving and structuring experiment data from W&B.
All metric data is stored in W&B; no local pickle files are needed.

Usage:
    from handle_data import get_runs, get_config, load_run_as_dataframe

    for run_id in get_runs():
        df = load_run_as_dataframe(run_id)
        ...
"""

import warnings
import wandb
import numpy as np
import pandas as pd
from config import base_config
from utils import smooth, compute_relative_change, first_convergence_index

# NOTE: all methods are untested

# W&B API client — reads credentials from the environment / ~/.netrc
_api: wandb.Api = wandb.Api()

# Project path in W&B: "entity/project_name"
PROJECT: str = f"{base_config['wandb_entity']}/{base_config['project_name']}"

_METRICS: tuple[str, ...] = ("return", "ep_len", "policy_loss", "value_loss")


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

    Because ppo_clip() emits ``{agent_name}/timestep`` in the *same*
    wandb.log() call as the four per-environment metrics, the actual training
    timestep is available on every history row that contains metric data.
    This function reads it directly from the same row and emits it as the
    ``training_timestep`` column, enabling downstream convergence analysis
    without any interpolation.

    Args:
        run_id:      W&B run ID string.
        max_samples: Maximum number of history rows to download. W&B
                     sub-samples history for large runs; increase this
                     if curves appear coarse.

    Returns:
        DataFrame with columns:
            run_id, agent_name, env, metric, value, step, training_timestep

        ``training_timestep`` is NaN on rows where the agent's timestep
        column is absent (e.g. IDM training rows).
    """
    run = _api.run(f"{PROJECT}/{run_id}")
    history: pd.DataFrame = run.history(samples=max_samples)

    # Pre-index all metric columns to avoid repeated string splitting.
    metric_cols: dict[str, tuple[str, str, str]] = {}
    for col in history.columns:
        parts: list[str] = col.split("/")
        if len(parts) == 3:
            agent_nm, env_nm, metric_nm = parts
            if metric_nm in _METRICS:
                metric_cols[col] = (agent_nm, env_nm, metric_nm)

    rows: list[dict] = []
    for _, row in history.iterrows():
        for col, (agent_nm, env_nm, metric_nm) in metric_cols.items():
            val = row.get(col)
            if pd.isna(val):
                continue
            # The training timestep is logged in the same wandb.log() call.
            ts_col: str = f"{agent_nm}/timestep"
            raw_ts = row.get(ts_col)
            training_ts: float = float(raw_ts) if not pd.isna(raw_ts) else float("nan")
            rows.append({
                "run_id":            run_id,
                "agent_name":        agent_nm,
                "env":               env_nm,
                "metric":            metric_nm,
                "value":             float(val),
                "step":              int(row["_step"]),
                "training_timestep": training_ts,
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


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

def compute_convergence_step(
    df: pd.DataFrame,
    agent_name: str,
    env_name: str,
    total_timesteps: int,
    threshold: float = 0.15,
) -> int:
    """Return the training timestep at which (agent_name, env_name) converged.

    Convergence is defined as the maximum over all four metrics
    (return, ep_len, policy_loss, value_loss) of the first training timestep
    where the relative change between consecutive smoothed logged values drops
    below ``threshold`` and stays there (i.e. the timestep after the last
    violation of the criterion).

    If a metric never converges, ``total_timesteps`` is used for that metric.
    The maximum across metrics is therefore always well-defined.

    Args:
        df:               Tidy DataFrame from load_run_as_dataframe(), which
                          must include the ``training_timestep`` column.
        agent_name:       Agent name prefix (e.g. "KingWorld_baseline").
        env_name:         Short env name as used in metric keys
                          (e.g. "KingWorld-v0").
        total_timesteps:  Fallback value used for any metric that never
                          converges.
        threshold:        Relative change threshold (e.g. 0.15 = 15%).

    Returns:
        Integer training timestep.
    """
    convergence_timesteps: list[int] = []

    for metric in _METRICS:
        sub: pd.DataFrame = (
            df[
                (df["agent_name"] == agent_name)
                & (df["env"] == env_name)
                & (df["metric"] == metric)
            ]
            .dropna(subset=["training_timestep"])
            .sort_values("training_timestep")
            .reset_index(drop=True)
        )

        if len(sub) < 2:
            # Not enough data points to compute relative changes.
            convergence_timesteps.append(total_timesteps)
            continue

        rel_changes: np.ndarray = compute_relative_change(sub["value"].tolist())
        idx: int | None = first_convergence_index(rel_changes, threshold)

        if idx is None:
            convergence_timesteps.append(total_timesteps)
        else:
            convergence_timesteps.append(int(sub["training_timestep"].iloc[idx]))

    return max(convergence_timesteps)


def get_min_convergence_step(
    run_id: str,
    training_agent_names: list[str],
    training_env_names: list[str],
    total_timesteps: int,
    threshold: float = 0.15,
    max_samples: int = 10_000,
) -> int:
    """Return the minimum convergence timestep across all training agents.

    For each baseline agent, convergence is the max over 4 metrics for its
    single training environment (inferred from the agent name by stripping the
    ``_baseline`` suffix and appending ``-v0``).  For the ASA agent
    (``"asa_agent"``), convergence is the max over all (env, metric) pairs
    across every training environment in which it has logged data.

    The minimum of those per-agent convergence timesteps is T_transfer.
    Falls back to ``total_timesteps`` if no agent converged on any metric
    (i.e. all per-agent values equal ``total_timesteps``), which is a safe
    and interpretable default — transfer runs get the full training budget.

    Args:
        run_id:                W&B run ID.
        training_agent_names:  Agent name strings as used in W&B metric keys
                               (e.g. ["KingWorld_baseline", "asa_agent"]).
        training_env_names:    Short env name strings without the
                               ``chess_env/`` prefix
                               (e.g. ["KingWorld-v0", "KnightWorld-v0"]).
        total_timesteps:       Fallback value for non-converging metrics /
                               agents.
        threshold:             Relative change threshold passed to
                               compute_convergence_step().
        max_samples:           Passed to load_run_as_dataframe().

    Returns:
        Integer training timestep to use as T_transfer.
    """
    df: pd.DataFrame = load_run_as_dataframe(run_id, max_samples)

    agent_convergences: list[int] = []

    for agent_name in training_agent_names:
        if agent_name == "asa_agent":
            # Only consider envs where this agent has actually logged data,
            # so that training envs outside asa_envs don't artificially inflate
            # the convergence estimate to total_timesteps.
            asa_envs_with_data: list[str] = [
                env_name for env_name in training_env_names
                if not df[
                    (df["agent_name"] == agent_name)
                    & (df["env"] == env_name)
                ].empty
            ]
            if not asa_envs_with_data:
                agent_convergences.append(total_timesteps)
            else:
                env_conv: list[int] = [
                    compute_convergence_step(
                        df, agent_name, env_name, total_timesteps, threshold
                    )
                    for env_name in asa_envs_with_data
                ]
                agent_convergences.append(max(env_conv))
        else:
            # Infer the single env this baseline agent trained in.
            # Naming convention: "{env_tag}_baseline" where
            # env_tag = env_name.replace("chess_env/", "").replace("-v0", "")
            # So reversing: env_name = agent_name[:-len("_baseline")] + "-v0"
            env_name: str = agent_name.replace("_baseline", "") + "-v0"
            agent_convergences.append(
                compute_convergence_step(
                    df, agent_name, env_name, total_timesteps, threshold
                )
            )

    if not agent_convergences:
        return total_timesteps

    result: int = min(agent_convergences)

    if result == total_timesteps:
        warnings.warn(
            "get_min_convergence_step: no training agent converged on any "
            f"metric within {total_timesteps} timesteps. Falling back to "
            f"T_transfer = {total_timesteps}.",
            stacklevel=2,
        )

    return result


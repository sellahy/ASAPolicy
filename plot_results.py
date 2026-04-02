"""plot_results.py

Generate comparison figures from W&B run data.

Usage (single run):
    python plot_results.py --run_id <wandb_run_id> [--save_dir figures/]

Usage (hyperparameter sensitivity across a sweep):
    python plot_results.py --run_id <any_run_id> --hyperparams gamma ppo_lr latent_action_dim
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from handle_data import (
    get_all_envs,
    get_config,
    get_runs,
    load_run_as_dataframe,
)


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def smooth(values: list[float], window: int = 11) -> np.ndarray:
    """Centered moving average with truncated windows at the edges.

    For each position i, averages over the largest centered window that fits
    within the array bounds, up to the given window size. Boundary points use
    a smaller window rather than being padded with zeros.

    For even-sized windows, floor(window/2) points are taken to the left and
    (window - floor(window/2) - 1) to the right, so interior points always
    average exactly `window` values.

    Example:
        smooth([1, 2, 3, 4, 5], window=3) -> [1.5, 2.0, 3.0, 4.0, 4.5]
          - i=0: avg([1, 2])    = 1.5  (only 1 right neighbour available)
          - i=1: avg([1, 2, 3]) = 2.0  (full window)
          - i=2: avg([2, 3, 4]) = 3.0  (full window)
          - i=3: avg([3, 4, 5]) = 4.0  (full window)
          - i=4: avg([4, 5])    = 4.5  (only 1 left neighbour available)

    Args:
        values: Sequence of scalar values to smooth.
        window: Total window size. Should be a positive odd integer for
                symmetric smoothing; even values are supported but produce a
                slightly left-biased window.

    Returns:
        np.ndarray of the same length as values.
    """
    arr: np.ndarray = np.array(values, dtype=float)
    n: int          = len(arr)
    half_left: int  = window // 2
    half_right: int = window - half_left - 1
    smoothed: np.ndarray = np.empty(n)
    for i in range(n):
        start: int      = max(0, i - half_left)
        end: int        = min(n - 1, i + half_right)
        smoothed[i]     = arr[start : end + 1].mean()
    return smoothed


# ---------------------------------------------------------------------------
# Shared bar chart helper
# ---------------------------------------------------------------------------

def _grouped_bar_chart(data: pd.DataFrame, metric_label: str,
                       title: str, filename: str) -> None:
    """Generic grouped bar chart with one group per env and one bar per agent.

    Args:
        data:         DataFrame with columns [agent_name, env, mean, sem].
        metric_label: Y-axis label.
        title:        Figure title.
        filename:     Full path to save the figure.
    """
    agents: np.ndarray = data["agent_name"].unique()
    envs: np.ndarray   = data["env"].unique()
    x: np.ndarray      = np.arange(len(envs))
    width: float       = 0.8 / max(len(agents), 1)
    colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, max(len(agents), 1)))

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(envs)), 5))
    for i, (agent, color) in enumerate(zip(agents, colors)):
        sub: pd.DataFrame = data[data["agent_name"] == agent].set_index("env").reindex(envs)
        offset: float = (i - len(agents) / 2 + 0.5) * width
        ax.bar(
            x + offset, sub["mean"], width,
            yerr=sub["sem"], capsize=4,
            label=agent, color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=30, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(df: pd.DataFrame, envs: list[str],
                         save_dir: str = "figures") -> None:
    """Smoothed return vs. training step, one subplot per environment.

    Error bands (±1 standard error) are drawn when multiple seeds are available.

    Args:
        df:       Tidy DataFrame from load_run_as_dataframe().
        envs:     List of environment names to plot.
        save_dir: Directory to save the figure.
    """
    n_cols: int = 4
    n_rows: int = -(-len(envs) // n_cols)  # ceiling division
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    ret_df: pd.DataFrame = df[df["metric"] == "return"]
    agents: np.ndarray   = ret_df["agent_name"].unique()
    colors: np.ndarray   = plt.cm.tab10(np.linspace(0, 1, max(len(agents), 1)))
    agent_color: dict[str, np.ndarray] = dict(zip(agents, colors))

    for idx, env in enumerate(envs):
        ax = axes[idx // n_cols][idx % n_cols]
        env_df: pd.DataFrame = ret_df[ret_df["env"] == env].sort_values("step")

        for agent in agents:
            ag_df: pd.DataFrame = env_df[env_df["agent_name"] == agent]
            if ag_df.empty:
                continue

            # Group by step across seeds, compute mean ± SE
            grouped: pd.DataFrame = (
                ag_df.groupby("step")["value"]
                .agg(mean="mean", sem="sem")
                .reset_index()
            )
            smoothed_mean: np.ndarray = smooth(grouped["mean"].tolist())
            ax.plot(grouped["step"], smoothed_mean,
                    label=agent, color=agent_color[agent])
            ax.fill_between(
                grouped["step"],
                smoothed_mean - grouped["sem"].values,
                smoothed_mean + grouped["sem"].values,
                alpha=0.2, color=agent_color[agent]
            )

        ax.set_title(env)
        ax.set_xlabel("Step")
        ax.set_ylabel("Smoothed return")
        ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(len(envs), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/learning_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Jumpstart bar chart
# ---------------------------------------------------------------------------

def plot_jumpstart(df: pd.DataFrame, K: int = 100,
                   save_dir: str = "figures") -> None:
    """Bar chart: mean return over first K eval events per (agent, env).

    Args:
        df:       Tidy DataFrame from load_run_as_dataframe().
        K:        Number of initial evaluation events to average over.
        save_dir: Directory to save the figure.
    """
    ret_df: pd.DataFrame = df[df["metric"] == "return"]
    early: pd.DataFrame = (
        ret_df.sort_values("step")
        .groupby(["run_id", "agent_name", "env"])
        .head(K)
        .groupby(["agent_name", "env"])["value"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )
    _grouped_bar_chart(
        early,
        metric_label=f"Mean return (first {K} evals)",
        title="Jumpstart Performance",
        filename=f"{save_dir}/jumpstart.png"
    )


# ---------------------------------------------------------------------------
# Plot 3 — Asymptotic performance bar chart
# ---------------------------------------------------------------------------

def plot_asymptotic(df: pd.DataFrame, K: int = 100,
                    save_dir: str = "figures") -> None:
    """Bar chart: mean return over last K eval events per (agent, env).

    Args:
        df:       Tidy DataFrame from load_run_as_dataframe().
        K:        Number of final evaluation events to average over.
        save_dir: Directory to save the figure.
    """
    ret_df: pd.DataFrame = df[df["metric"] == "return"]
    late: pd.DataFrame = (
        ret_df.sort_values("step")
        .groupby(["run_id", "agent_name", "env"])
        .tail(K)
        .groupby(["agent_name", "env"])["value"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )
    _grouped_bar_chart(
        late,
        metric_label=f"Mean return (last {K} evals)",
        title="Asymptotic Performance",
        filename=f"{save_dir}/asymptotic.png"
    )


# ---------------------------------------------------------------------------
# Plot 4 — Return convergence speed
# ---------------------------------------------------------------------------

def _convergence_step_return(group: pd.DataFrame,
                              threshold: float = 0.8) -> float:
    """First step where smoothed return >= threshold * asymptotic mean.

    Args:
        group:     DataFrame slice for one (run_id, agent_name, env) group.
        threshold: Fraction of the asymptote to use as the convergence target.

    Returns:
        Step index at convergence, or NaN if never reached.
    """
    group = group.sort_values("step")
    asymptote: float = float(group["value"].iloc[-100:].mean())
    target: float    = threshold * asymptote
    smoothed: np.ndarray = smooth(group["value"].tolist())
    crossed: np.ndarray  = np.where(smoothed >= target)[0]
    return float(group["step"].iloc[crossed[0]]) if len(crossed) > 0 else float("nan")


def plot_return_convergence_speed(df: pd.DataFrame, threshold: float = 0.8,
                                  save_dir: str = "figures") -> None:
    """Bar chart: step at which return first reaches threshold * asymptote.

    Lower is faster / better.

    Args:
        df:        Tidy DataFrame from load_run_as_dataframe().
        threshold: Convergence threshold as a fraction of the asymptote.
        save_dir:  Directory to save the figure.
    """
    ret_df: pd.DataFrame = df[df["metric"] == "return"]
    conv: pd.DataFrame = (
        ret_df.groupby(["run_id", "agent_name", "env"])
        .apply(_convergence_step_return, threshold=threshold)
        .reset_index(name="convergence_step")
        .groupby(["agent_name", "env"])["convergence_step"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )
    _grouped_bar_chart(
        conv,
        metric_label="Step to convergence (lower = faster)",
        title=f"Return Convergence Speed (threshold={threshold})",
        filename=f"{save_dir}/return_convergence_speed.png"
    )


# ---------------------------------------------------------------------------
# Plot 5 — Loss convergence speed
# ---------------------------------------------------------------------------

def _convergence_step_loss(group: pd.DataFrame,
                            threshold: float = 0.8) -> float:
    """First step where smoothed loss <= threshold * initial mean loss.

    Loss convergence direction is opposite to return convergence: we look for
    the loss to *decrease* to the threshold level.

    Args:
        group:     DataFrame slice for one (run_id, agent_name, env) group.
        threshold: Target as a fraction of the initial loss.

    Returns:
        Step index at convergence, or NaN if never reached.
    """
    group = group.sort_values("step")
    initial: float   = float(group["value"].iloc[:10].mean())  # average of first 10 steps
    target: float    = threshold * initial
    smoothed: np.ndarray = smooth(group["value"].tolist())
    crossed: np.ndarray  = np.where(smoothed <= target)[0]
    return float(group["step"].iloc[crossed[0]]) if len(crossed) > 0 else float("nan")


def plot_loss_convergence_speed(df: pd.DataFrame, threshold: float = 0.8,
                                save_dir: str = "figures") -> None:
    """Bar charts: steps at which policy_loss and value_loss first drop to
    threshold * initial loss. Produces two separate figures.

    Args:
        df:        Tidy DataFrame from load_run_as_dataframe().
        threshold: Convergence threshold as a fraction of the initial loss.
        save_dir:  Directory to save figures.
    """
    for loss_type in ("policy_loss", "value_loss"):
        loss_df: pd.DataFrame = df[df["metric"] == loss_type]
        conv: pd.DataFrame = (
            loss_df.groupby(["run_id", "agent_name", "env"])
            .apply(_convergence_step_loss, threshold=threshold)
            .reset_index(name="convergence_step")
            .groupby(["agent_name", "env"])["convergence_step"]
            .agg(mean="mean", sem="sem")
            .reset_index()
        )
        _grouped_bar_chart(
            conv,
            metric_label="Step to convergence (lower = faster)",
            title=(
                f"{loss_type.replace('_', ' ').title()} Convergence Speed "
                f"(threshold={threshold})"
            ),
            filename=f"{save_dir}/{loss_type}_convergence_speed.png"
        )


# ---------------------------------------------------------------------------
# Plot 6 — Hyperparameter sensitivity
# ---------------------------------------------------------------------------

def plot_hyperparam_sensitivity(df: pd.DataFrame,
                                run_configs: dict[str, dict],
                                metric: str,
                                hyperparams: list[str],
                                env: str,
                                agent_name: str,
                                save_dir: str = "figures") -> None:
    """One subplot per hyperparameter showing metric mean ± SE across sweep runs.

    Args:
        df:           Tidy DataFrame across multiple runs.
        run_configs:  Dict mapping run_id -> config dict (from get_config()).
        metric:       Metric name to plot (e.g. "return").
        hyperparams:  List of hyperparameter keys to sweep across.
        env:          Environment name to filter on.
        agent_name:   Agent name to filter on.
        save_dir:     Directory to save the figure.
    """
    sub: pd.DataFrame = df[
        (df["env"] == env)
        & (df["agent_name"] == agent_name)
        & (df["metric"] == metric)
    ].copy()

    # Attach config values to each row
    for hp in hyperparams:
        sub[hp] = sub["run_id"].map(
            lambda rid, h=hp: run_configs.get(rid, {}).get(h, float("nan"))
        )

    fig, axes = plt.subplots(1, len(hyperparams),
                              figsize=(5 * len(hyperparams), 4))
    if len(hyperparams) == 1:
        axes = [axes]

    for ax, hp in zip(axes, hyperparams):
        grouped: pd.DataFrame = (
            sub.groupby(hp)["value"]
            .agg(mean="mean", sem="sem")
            .reset_index()
            .sort_values(hp)
        )
        ax.errorbar(grouped[hp], grouped["mean"], yerr=grouped["sem"],
                    marker="o", capsize=4, linestyle="-")
        ax.set_xlabel(hp)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {hp}\n({env}, {agent_name})")

    fig.tight_layout()
    fig.savefig(
        f"{save_dir}/hyperparam_{env}_{agent_name}_{metric}.png", dpi=150
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from W&B run data."
    )
    parser.add_argument(
        "--run_id", type=str, required=True,
        help="W&B run ID to plot."
    )
    parser.add_argument(
        "--save_dir", type=str, default="figures",
        help="Directory to save plot images."
    )
    parser.add_argument(
        "--hyperparams", nargs="*", default=[],
        help="Hyperparameter names for sensitivity plots. "
             "Only useful when the run is part of a W&B sweep."
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cfg: dict    = get_config(args.run_id)
    K: int       = int(cfg.get("jumpstart_K", 100))
    K_late: int  = int(cfg.get("asymptote_K", 100))
    thresh: float = float(cfg.get("convergence_threshold", 0.8))

    print(f"Downloading run history for {args.run_id} ...")
    df: pd.DataFrame    = load_run_as_dataframe(args.run_id)
    envs: list[str]     = get_all_envs(args.run_id)

    print("Generating plots ...")
    plot_learning_curves(df, envs, save_dir=args.save_dir)
    plot_jumpstart(df, K=K, save_dir=args.save_dir)
    plot_asymptotic(df, K=K_late, save_dir=args.save_dir)
    plot_return_convergence_speed(df, threshold=thresh, save_dir=args.save_dir)
    plot_loss_convergence_speed(df, threshold=thresh, save_dir=args.save_dir)

    if args.hyperparams:
        run_configs: dict[str, dict] = {rid: get_config(rid) for rid in get_runs()}
        for env in envs:
            for agent in df["agent_name"].unique():
                plot_hyperparam_sensitivity(
                    df, run_configs, metric="return",
                    hyperparams=args.hyperparams,
                    env=env, agent_name=str(agent),
                    save_dir=args.save_dir
                )

    print(f"Plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()

"""Sim-to-real evaluation for the actuator model.

Loads all .npz trajectory files from a given directory and produces four
diagnostic plots comparing simulated vs. real robot observations:

    1. Velocity comparison  - commanded, real, and sim velocities overlaid
    2. Performance metrics  - per-trajectory RMSE bar charts for X/Y velocity
    3. Velocity deviation   - (sim - real) velocity error over time
    4. Peg position         - sim vs. real positions and their deviation

Velocity error metrics (MSE/RMSE) are saved to sim2real_evaluation_metrics.json
in the same directory as the figures.

Figures are saved next to the data when running headless (Agg backend),
or displayed interactively otherwise.

Example usage:
    # Combined subplot figures (default, matches the original layout)
    python sim2real_evaluation.py data/evaluation/new/from_new_model/seed1

    # One PNG per trajectory panel — useful when there are many trajectories
    python sim2real_evaluation.py data/evaluation/new/from_new_model/seed1 --separate
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_or_show(fig, name, output_dir):
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = output_dir / name
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def _traj_slug(t):
    """Stable filename component for a trajectory (stem of its source file)."""
    return t["filename"].stem


def load_trajectories(filenames, skip_first_steps=0):
    """Load .npz trajectory files and return a list of dicts with extracted arrays.

    Each dict contains time steps, commanded actions, real/sim velocities, and
    real/sim peg positions extracted from observations_real and observations_sim.
    Files that fail to load are skipped with a warning.

    Args:
        filenames: Iterable of Path objects pointing to .npz files.
        skip_first_steps: Number of leading steps to drop from every trajectory.

    Returns:
        List of dicts, one per successfully loaded file.
    """
    trajectories = []
    for i, filename in enumerate(filenames):
        try:
            data = np.load(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        obs_real = data["observations_real"]
        obs_sim = data["observations_sim"]
        actions = data["actions"]

        n = obs_real.shape[0]
        if skip_first_steps >= n:
            print(
                f"Warning: {filename.name} has only {n} steps, "
                f"shorter than --skip-first-steps={skip_first_steps}; skipping."
            )
            continue
        s = skip_first_steps

        trajectories.append({
            "index": i,
            "filename": filename,
            "time_steps": np.arange(n - s),
            "action_x": actions[s:, 0],
            "action_y": actions[s:, 1],
            # Velocities at indices [2, 3]
            "vel_real_x": obs_real[s:, 2],
            "vel_real_y": obs_real[s:, 3],
            "vel_sim_x": obs_sim[s:, 2],
            "vel_sim_y": obs_sim[s:, 3],
            # Own peg positions at indices [0, 1]
            "pos_real_x": obs_real[s:, 0],
            "pos_real_y": obs_real[s:, 1],
            "pos_sim_x": obs_sim[s:, 0],
            "pos_sim_y": obs_sim[s:, 1],
        })

    return trajectories


def calculate_metrics(trajectories):
    """Calculate MSE and RMSE between sim and real velocities for each trajectory.

    Prints per-trajectory and mean error metrics to stdout.

    Args:
        trajectories: List of trajectory dicts as returned by load_trajectories.

    Returns:
        Dict mapping trajectory index to {"mse_x", "mse_y", "rmse_x", "rmse_y"}.
    """
    metrics = {}
    print("Calculating Velocity Error Metrics:")
    for t in trajectories:
        i = t["index"]
        mse_x = np.mean((t["vel_real_x"] - t["vel_sim_x"]) ** 2)
        mse_y = np.mean((t["vel_real_y"] - t["vel_sim_y"]) ** 2)
        rmse_x = np.sqrt(mse_x)
        rmse_y = np.sqrt(mse_y)

        print(f"\n  Trajectory {i+1} ({t['filename'].name}):")
        print(f"    MSE  (Vel X): {mse_x:.6f}   RMSE (Vel X): {rmse_x:.6f}")
        print(f"    MSE  (Vel Y): {mse_y:.6f}   RMSE (Vel Y): {rmse_y:.6f}")

        metrics[i] = {"mse_x": mse_x, "mse_y": mse_y, "rmse_x": rmse_x, "rmse_y": rmse_y}

    return metrics


def save_metrics(metrics, trajectories, output_dir):
    """Save computed velocity error metrics to a JSON file.

    Writes per-trajectory MSE/RMSE values plus mean RMSE across all trajectories
    to sim2real_evaluation_metrics.json in output_dir.

    Args:
        metrics: Dict as returned by calculate_metrics.
        trajectories: List of trajectory dicts as returned by load_trajectories.
        output_dir: Path to the directory where the JSON file is saved.
    """
    rmse_x_vals = [metrics[i]["rmse_x"] for i in sorted(metrics)]
    rmse_y_vals = [metrics[i]["rmse_y"] for i in sorted(metrics)]

    output = {
        "trajectories": {
            trajectories[i]["filename"].name: {k: float(v) for k, v in metrics[i].items()} for i in sorted(metrics)
        },
        "mean_rmse_x": float(np.mean(rmse_x_vals)),
        "mean_rmse_y": float(np.mean(rmse_y_vals)),
    }

    output_path = output_dir / "sim2real_evaluation_metrics.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved metrics to {output_path}")


def _draw_velocity_comparison(ax, t):
    ts = t["time_steps"]
    ax.step(ts, t["action_x"], label="Commanded Vel X", color="gray", linestyle=":", alpha=0.8)
    ax.step(ts, t["action_y"], label="Commanded Vel Y", color="silver", linestyle=":", alpha=0.8)
    ax.plot(ts, t["vel_real_x"], label="Real Vel X", color="tab:blue", linestyle="-", linewidth=2)
    ax.plot(ts, t["vel_sim_x"], label="Sim Vel X", color="tab:cyan", linestyle="--", linewidth=2)
    ax.plot(ts, t["vel_real_y"], label="Real Vel Y", color="tab:orange", linestyle="-", linewidth=2)
    ax.plot(ts, t["vel_sim_y"], label="Sim Vel Y", color="tab:red", linestyle="--", linewidth=2)
    ax.set_title(f"Trajectory {t['index']+1}", fontsize=12)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Velocity (m/s)")
    ax.grid(True, linestyle=":", alpha=0.6)


def plot_velocity_comparison(trajectories, output_dir, separate=False):
    """Plot commanded, real, and sim velocities overlaid for each trajectory."""
    if separate:
        for t in trajectories:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            fig.suptitle("Sim-to-Real Actuator Model Validation", fontsize=14, weight="bold")
            _draw_velocity_comparison(ax, t)
            ax.legend(loc="best", fontsize=9)
            _save_or_show(
                fig,
                f"sim2real_evaluation_01_velocity_comparison_{_traj_slug(t)}.png",
                output_dir,
            )
            plt.close(fig)
        return

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Sim-to-Real Actuator Model Validation", fontsize=18, weight="bold")
    ax_flat = axes.flatten()

    for t in trajectories:
        _draw_velocity_comparison(ax_flat[t["index"]], t)

    handles, labels = ax_flat[0].get_legend_handles_labels()
    ax_flat[-1].legend(handles, labels, loc="center", fontsize=12)
    ax_flat[-1].axis("off")

    _save_or_show(fig, "sim2real_evaluation_01_velocity_comparison.png", output_dir)
    plt.close(fig)


def _draw_rmse_bars(ax, x, labels, vals, color, title):
    bars = ax.bar(x, vals, 0.6, color=color, alpha=0.8)
    mean_val = float(np.mean(vals))
    ax.axhline(mean_val, color="tab:red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.4f}")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0, ha="right" if len(labels) > 6 else "center")
    ax.set_ylabel("RMSE (m/s)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return mean_val


def plot_performance_metrics(metrics, output_dir, separate=False):
    """Plot bar charts of RMSE per trajectory for X and Y velocities."""
    labels = [f"Traj {i+1}" for i in sorted(metrics)]
    rmse_x_vals = [metrics[i]["rmse_x"] for i in sorted(metrics)]
    rmse_y_vals = [metrics[i]["rmse_y"] for i in sorted(metrics)]
    x = np.arange(len(labels))
    fig_width = max(8, 0.6 * len(labels) + 4)

    if separate:
        fig_x, ax_x = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)
        fig_x.suptitle("Actuator Model Performance Metrics (RMSE)", fontsize=14, weight="bold")
        mean_rmse_x = _draw_rmse_bars(ax_x, x, labels, rmse_x_vals, "tab:blue", "RMSE — Velocity X")
        _save_or_show(fig_x, "sim2real_evaluation_02_performance_metrics_rmse_x.png", output_dir)
        plt.close(fig_x)

        fig_y, ax_y = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)
        fig_y.suptitle("Actuator Model Performance Metrics (RMSE)", fontsize=14, weight="bold")
        mean_rmse_y = _draw_rmse_bars(ax_y, x, labels, rmse_y_vals, "tab:orange", "RMSE — Velocity Y")
        _save_or_show(fig_y, "sim2real_evaluation_02_performance_metrics_rmse_y.png", output_dir)
        plt.close(fig_y)

        print(f"\n  Mean RMSE X: {mean_rmse_x:.6f}   Mean RMSE Y: {mean_rmse_y:.6f}")
        return

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(max(12, fig_width * 2), 5), constrained_layout=True)
    fig.suptitle("Actuator Model Performance Metrics (RMSE)", fontsize=16, weight="bold")
    mean_rmse_x = _draw_rmse_bars(ax1, x, labels, rmse_x_vals, "tab:blue", "RMSE — Velocity X")
    mean_rmse_y = _draw_rmse_bars(ax2, x, labels, rmse_y_vals, "tab:orange", "RMSE — Velocity Y")
    print(f"\n  Mean RMSE X: {mean_rmse_x:.6f}   Mean RMSE Y: {mean_rmse_y:.6f}")

    _save_or_show(fig, "sim2real_evaluation_02_performance_metrics.png", output_dir)
    plt.close(fig)


def _draw_velocity_deviation(ax, t):
    ts = t["time_steps"]
    dev_x = t["vel_sim_x"] - t["vel_real_x"]
    dev_y = t["vel_sim_y"] - t["vel_real_y"]
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.plot(ts, dev_x, label="Dev Vel X (sim−real)", color="tab:blue", linewidth=1.5)
    ax.plot(ts, dev_y, label="Dev Vel Y (sim−real)", color="tab:orange", linewidth=1.5)
    ax.set_title(f"Trajectory {t['index']+1}", fontsize=12)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Velocity Error (sim − real) (m/s)")
    ax.grid(True, linestyle=":", alpha=0.6)


def plot_velocity_deviation(trajectories, output_dir, separate=False):
    """Plot velocity deviation (sim - real) over time for each trajectory."""
    if separate:
        for t in trajectories:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            fig.suptitle("Sim-to-Real Velocity Deviation", fontsize=14, weight="bold")
            _draw_velocity_deviation(ax, t)
            ax.legend(loc="best", fontsize=9)
            _save_or_show(
                fig,
                f"sim2real_evaluation_03_velocity_deviation_{_traj_slug(t)}.png",
                output_dir,
            )
            plt.close(fig)
        return

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Sim-to-Real Velocity Deviation", fontsize=18, weight="bold")
    ax_flat = axes.flatten()

    for t in trajectories:
        _draw_velocity_deviation(ax_flat[t["index"]], t)

    handles, labels = ax_flat[0].get_legend_handles_labels()
    ax_flat[-1].legend(handles, labels, loc="center", fontsize=12)
    ax_flat[-1].axis("off")

    _save_or_show(fig, "sim2real_evaluation_03_velocity_deviation.png", output_dir)
    plt.close(fig)


def _draw_peg_position_comparison(ax, t):
    ts = t["time_steps"]
    ax.plot(ts, t["pos_real_x"], label="Real Pos X", color="tab:blue", linestyle="-", linewidth=2)
    ax.plot(ts, t["pos_sim_x"], label="Sim Pos X", color="tab:cyan", linestyle="--", linewidth=2)
    ax.plot(ts, t["pos_real_y"], label="Real Pos Y", color="tab:orange", linestyle="-", linewidth=2)
    ax.plot(ts, t["pos_sim_y"], label="Sim Pos Y", color="tab:red", linestyle="--", linewidth=2)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Position (m)")
    ax.grid(True, linestyle=":", alpha=0.6)


def _draw_peg_position_deviation(ax, t):
    ts = t["time_steps"]
    dev_pos_x = t["pos_sim_x"] - t["pos_real_x"]
    dev_pos_y = t["pos_sim_y"] - t["pos_real_y"]
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.plot(ts, dev_pos_x, label="Dev Pos X (sim−real)", color="tab:blue", linewidth=1.5)
    ax.plot(ts, dev_pos_y, label="Dev Pos Y (sim−real)", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Position Error (sim − real) (m)")
    ax.grid(True, linestyle=":", alpha=0.6)


def plot_peg_position(trajectories, output_dir, separate=False):
    """Plot peg position sim vs real and position deviation for each trajectory."""
    if separate:
        for t in trajectories:
            slug = _traj_slug(t)

            fig_cmp, ax_cmp = plt.subplots(figsize=(8, 5), constrained_layout=True)
            fig_cmp.suptitle(
                f"Sim-to-Real Peg Position Comparison — Trajectory {t['index']+1}", fontsize=13, weight="bold"
            )
            _draw_peg_position_comparison(ax_cmp, t)
            ax_cmp.legend(loc="best", fontsize=9)
            _save_or_show(fig_cmp, f"sim2real_evaluation_04_peg_position_{slug}.png", output_dir)
            plt.close(fig_cmp)

            fig_dev, ax_dev = plt.subplots(figsize=(8, 5), constrained_layout=True)
            fig_dev.suptitle(f"Peg Position Deviation — Trajectory {t['index']+1}", fontsize=13, weight="bold")
            _draw_peg_position_deviation(ax_dev, t)
            ax_dev.legend(loc="best", fontsize=9)
            _save_or_show(fig_dev, f"sim2real_evaluation_04_peg_position_deviation_{slug}.png", output_dir)
            plt.close(fig_dev)
        return

    # 6 rows = 3 rows for position comparison + 3 rows for position deviation
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 24), constrained_layout=True)
    fig.suptitle("Sim-to-Real Peg Position", fontsize=18, weight="bold")

    top_axes = axes[:3].flatten()
    bot_axes = axes[3:].flatten()

    top_axes[0].set_title("Sim-to-Real Peg Position Comparison — Trajectory 1", fontsize=11)
    bot_axes[0].set_title("Peg Position Deviation — Trajectory 1", fontsize=11)

    for t in trajectories:
        idx = t["index"]
        _draw_peg_position_comparison(top_axes[idx], t)
        _draw_peg_position_deviation(bot_axes[idx], t)
        if idx != 0:
            top_axes[idx].set_title(f"Trajectory {idx+1}", fontsize=11)
            bot_axes[idx].set_title(f"Trajectory {idx+1}", fontsize=11)

    handles_top, labels_top = top_axes[0].get_legend_handles_labels()
    top_axes[-1].legend(handles_top, labels_top, loc="center", fontsize=12)
    top_axes[-1].axis("off")

    handles_bot, labels_bot = bot_axes[0].get_legend_handles_labels()
    bot_axes[-1].legend(handles_bot, labels_bot, loc="center", fontsize=12)
    bot_axes[-1].axis("off")

    _save_or_show(fig, "sim2real_evaluation_04_peg_position.png", output_dir)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sim2real trajectories from .npz files.")
    parser.add_argument("data_dir", type=Path, help="Folder containing .npz trajectory files")
    parser.add_argument(
        "--separate",
        action="store_true",
        help=(
            "Emit each per-trajectory panel (and each RMSE bar chart) as its own PNG "
            "instead of one combined subplot figure."
        ),
    )
    parser.add_argument(
        "--skip-first-steps",
        type=int,
        default=80,
        help="Ignore the first N steps of each trajectory in all analysis and plots.",
    )
    args = parser.parse_args()

    filenames = sorted(args.data_dir.glob("*.npz"))
    if not filenames:
        raise SystemExit(f"No .npz files found in {args.data_dir}")

    trajectories = load_trajectories(filenames, skip_first_steps=args.skip_first_steps)
    if not trajectories:
        raise SystemExit("No trajectories left after --skip-first-steps filtering.")
    metrics = calculate_metrics(trajectories)
    save_metrics(metrics, trajectories, args.data_dir)
    plot_velocity_comparison(trajectories, args.data_dir, separate=args.separate)
    plot_performance_metrics(metrics, args.data_dir, separate=args.separate)
    plot_velocity_deviation(trajectories, args.data_dir, separate=args.separate)
    plot_peg_position(trajectories, args.data_dir, separate=args.separate)

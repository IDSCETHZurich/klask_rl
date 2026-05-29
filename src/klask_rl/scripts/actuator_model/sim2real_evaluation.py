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


def grid_layout(n_slots):
    """Return (nrows, ncols) for laying out n_slots panels in an up-to-2-column grid."""
    ncols = 2 if n_slots > 1 else 1
    nrows = (n_slots + ncols - 1) // ncols
    return nrows, ncols


def save_figure(fig, name, output_dir, dpi=150):
    """Save fig to output_dir/name when running on Agg, otherwise show interactively."""
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = output_dir / name
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def make_combined_grid(
    n_data_panels,
    suptitle,
    *,
    suptitle_fontsize=18,
    fig_width_per_col=7.5,
    fig_height_per_row=4,
):
    """Create a figure whose grid holds n_data_panels + 1 panels (last one for the legend).

    Returns (fig, ax_flat). The caller draws data into ax_flat[0:n_data_panels],
    then calls finalize_combined_grid_legend(ax_flat, n_data_panels) to populate
    the legend slot and hide any trailing unused panels.
    """
    nrows, ncols = grid_layout(n_data_panels + 1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width_per_col * ncols, fig_height_per_row * nrows),
        constrained_layout=True,
    )
    fig.suptitle(suptitle, fontsize=suptitle_fontsize, weight="bold")
    ax_flat = np.atleast_1d(axes).flatten()
    return fig, ax_flat


def finalize_combined_grid_legend(ax_flat, n_data_panels, *, legend_fontsize=12):
    """Place a shared legend in the slot after the last data panel and hide any unused trailing slots."""
    handles, labels = ax_flat[0].get_legend_handles_labels()
    legend_ax = ax_flat[n_data_panels]
    legend_ax.legend(handles, labels, loc="center", fontsize=legend_fontsize)
    legend_ax.axis("off")
    for i in range(n_data_panels + 1, len(ax_flat)):
        ax_flat[i].axis("off")


def _traj_slug(t):
    """Stable filename component for a trajectory (stem of its source file)."""
    return t["filename"].stem


def load_trajectory_arrays(npz_path, skip_first_steps=0):
    """Load a single .npz trajectory file into a flat dict of float arrays.

    Conventions:
        observations_real / observations_sim: columns [pos_x, pos_y, vel_x, vel_y, ...]
        actions: columns [action_x, action_y]

    Args:
        npz_path: Path to the .npz file.
        skip_first_steps: Number of leading steps to drop.

    Returns:
        Dict with time_steps, action_x/y, vel_real_x/y, vel_sim_x/y, pos_real_x/y, pos_sim_x/y.

    Raises:
        ValueError: if the trajectory is shorter than skip_first_steps.
    """
    with np.load(npz_path) as data:
        obs_real = np.asarray(data["observations_real"], dtype=float)
        obs_sim = np.asarray(data["observations_sim"], dtype=float)
        actions = np.asarray(data["actions"], dtype=float)

    n = obs_real.shape[0]
    if skip_first_steps >= n:
        raise ValueError(f"trajectory has only {n} steps, shorter than skip_first_steps={skip_first_steps}")
    s = skip_first_steps

    return {
        "time_steps": np.arange(n - s),
        "action_x": actions[s:, 0],
        "action_y": actions[s:, 1],
        "vel_real_x": obs_real[s:, 2],
        "vel_real_y": obs_real[s:, 3],
        "vel_sim_x": obs_sim[s:, 2],
        "vel_sim_y": obs_sim[s:, 3],
        "pos_real_x": obs_real[s:, 0],
        "pos_real_y": obs_real[s:, 1],
        "pos_sim_x": obs_sim[s:, 0],
        "pos_sim_y": obs_sim[s:, 1],
    }


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
            arr = load_trajectory_arrays(filename, skip_first_steps=skip_first_steps)
        except ValueError as e:
            print(f"Warning: {filename.name}: {e}; skipping.")
            continue
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        trajectories.append({"index": i, "filename": filename, **arr})

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


def draw_velocity_comparison(
    ax,
    ts,
    action_x,
    action_y,
    real_x,
    real_y,
    sim_x,
    sim_y,
    *,
    sim_x_sigma=None,
    sim_y_sigma=None,
    title=None,
):
    """Draw commanded, real, and sim velocity overlays on a single axis.

    If sim_x_sigma / sim_y_sigma are provided, sim_x / sim_y are interpreted as
    cross-seed means and ±1σ shaded bands are drawn around them (and the sim
    line labels reflect that they are means).
    """
    has_sigma = sim_x_sigma is not None and sim_y_sigma is not None
    sim_x_label = "Sim Vel X (mean)" if has_sigma else "Sim Vel X"
    sim_y_label = "Sim Vel Y (mean)" if has_sigma else "Sim Vel Y"

    ax.step(ts, action_x, label="Commanded Vel X", color="tab:green", linestyle=":", alpha=0.8)
    ax.step(ts, action_y, label="Commanded Vel Y", color="tab:red", linestyle=":", alpha=0.8)
    ax.plot(ts, real_x, label="Real Vel X", color="limegreen", linestyle="-", linewidth=2)
    ax.plot(ts, sim_x, label=sim_x_label, color="darkgreen", linestyle="--", linewidth=2)
    if has_sigma:
        ax.fill_between(
            ts, sim_x - sim_x_sigma, sim_x + sim_x_sigma, color="tab:green", alpha=0.3, label="Sim Vel X ±1σ"
        )
    ax.plot(ts, real_y, label="Real Vel Y", color="tab:red", linestyle="-", linewidth=2)
    ax.plot(ts, sim_y, label=sim_y_label, color="darkred", linestyle="--", linewidth=2)
    if has_sigma:
        ax.fill_between(ts, sim_y - sim_y_sigma, sim_y + sim_y_sigma, color="tab:red", alpha=0.3, label="Sim Vel Y ±1σ")

    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Velocity (m/s)")
    ax.grid(True, linestyle=":", alpha=0.6)


def _draw_velocity_comparison(ax, t):
    draw_velocity_comparison(
        ax,
        t["time_steps"],
        t["action_x"],
        t["action_y"],
        t["vel_real_x"],
        t["vel_real_y"],
        t["vel_sim_x"],
        t["vel_sim_y"],
        title=f"Trajectory {t['index']+1}",
    )


def plot_velocity_comparison(trajectories, output_dir, separate=False):
    """Plot commanded, real, and sim velocities overlaid for each trajectory."""
    if separate:
        for t in trajectories:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            fig.suptitle("Sim-to-Real Actuator Model Validation", fontsize=14, weight="bold")
            _draw_velocity_comparison(ax, t)
            ax.legend(loc="best", fontsize=9)
            save_figure(
                fig,
                f"sim2real_evaluation_01_velocity_comparison_{_traj_slug(t)}.png",
                output_dir,
            )
            plt.close(fig)
        return

    n = len(trajectories)
    if n == 0:
        return
    fig, ax_flat = make_combined_grid(n, "Sim-to-Real Actuator Model Validation")

    for slot, t in enumerate(trajectories):
        _draw_velocity_comparison(ax_flat[slot], t)

    finalize_combined_grid_legend(ax_flat, n)

    save_figure(fig, "sim2real_evaluation_01_velocity_comparison.png", output_dir)
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
        save_figure(fig_x, "sim2real_evaluation_02_performance_metrics_rmse_x.png", output_dir)
        plt.close(fig_x)

        fig_y, ax_y = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)
        fig_y.suptitle("Actuator Model Performance Metrics (RMSE)", fontsize=14, weight="bold")
        mean_rmse_y = _draw_rmse_bars(ax_y, x, labels, rmse_y_vals, "tab:orange", "RMSE — Velocity Y")
        save_figure(fig_y, "sim2real_evaluation_02_performance_metrics_rmse_y.png", output_dir)
        plt.close(fig_y)

        print(f"\n  Mean RMSE X: {mean_rmse_x:.6f}   Mean RMSE Y: {mean_rmse_y:.6f}")
        return

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(max(12, fig_width * 2), 5), constrained_layout=True)
    fig.suptitle("Actuator Model Performance Metrics (RMSE)", fontsize=16, weight="bold")
    mean_rmse_x = _draw_rmse_bars(ax1, x, labels, rmse_x_vals, "tab:blue", "RMSE — Velocity X")
    mean_rmse_y = _draw_rmse_bars(ax2, x, labels, rmse_y_vals, "tab:orange", "RMSE — Velocity Y")
    print(f"\n  Mean RMSE X: {mean_rmse_x:.6f}   Mean RMSE Y: {mean_rmse_y:.6f}")

    save_figure(fig, "sim2real_evaluation_02_performance_metrics.png", output_dir)
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
            save_figure(
                fig,
                f"sim2real_evaluation_03_velocity_deviation_{_traj_slug(t)}.png",
                output_dir,
            )
            plt.close(fig)
        return

    n = len(trajectories)
    if n == 0:
        return
    fig, ax_flat = make_combined_grid(n, "Sim-to-Real Velocity Deviation")

    for slot, t in enumerate(trajectories):
        _draw_velocity_deviation(ax_flat[slot], t)

    finalize_combined_grid_legend(ax_flat, n)

    save_figure(fig, "sim2real_evaluation_03_velocity_deviation.png", output_dir)
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
            save_figure(fig_cmp, f"sim2real_evaluation_04_peg_position_{slug}.png", output_dir)
            plt.close(fig_cmp)

            fig_dev, ax_dev = plt.subplots(figsize=(8, 5), constrained_layout=True)
            fig_dev.suptitle(f"Peg Position Deviation — Trajectory {t['index']+1}", fontsize=13, weight="bold")
            _draw_peg_position_deviation(ax_dev, t)
            ax_dev.legend(loc="best", fontsize=9)
            save_figure(fig_dev, f"sim2real_evaluation_04_peg_position_deviation_{slug}.png", output_dir)
            plt.close(fig_dev)
        return

    n = len(trajectories)
    if n == 0:
        return
    # Each section holds n trajectory panels + 1 legend panel.
    section_rows, ncols = grid_layout(n + 1)
    fig, axes = plt.subplots(
        nrows=2 * section_rows, ncols=ncols, figsize=(7.5 * ncols, 4 * 2 * section_rows), constrained_layout=True
    )
    fig.suptitle("Sim-to-Real Peg Position", fontsize=18, weight="bold")

    axes = np.atleast_2d(axes)
    top_axes = axes[:section_rows].flatten()
    bot_axes = axes[section_rows:].flatten()

    for slot, t in enumerate(trajectories):
        _draw_peg_position_comparison(top_axes[slot], t)
        _draw_peg_position_deviation(bot_axes[slot], t)
        if slot == 0:
            top_axes[slot].set_title("Sim-to-Real Peg Position Comparison — Trajectory 1", fontsize=11)
            bot_axes[slot].set_title("Peg Position Deviation — Trajectory 1", fontsize=11)
        else:
            top_axes[slot].set_title(f"Trajectory {t['index']+1}", fontsize=11)
            bot_axes[slot].set_title(f"Trajectory {t['index']+1}", fontsize=11)

    handles_top, labels_top = top_axes[0].get_legend_handles_labels()
    top_axes[n].legend(handles_top, labels_top, loc="center", fontsize=12)
    top_axes[n].axis("off")

    handles_bot, labels_bot = bot_axes[0].get_legend_handles_labels()
    bot_axes[n].legend(handles_bot, labels_bot, loc="center", fontsize=12)
    bot_axes[n].axis("off")

    for i in range(n + 1, len(top_axes)):
        top_axes[i].axis("off")
        bot_axes[i].axis("off")

    save_figure(fig, "sim2real_evaluation_04_peg_position.png", output_dir)
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

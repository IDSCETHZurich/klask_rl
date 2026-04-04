from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent / "data/evaluation/new/from_new_model/seed1"
FILENAMES = [DATA_DIR / f"sim2real_trajectories_0{i}.npz" for i in range(1, 6)]


def _save_or_show(fig, name):
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = DATA_DIR / name
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def load_trajectories(filenames):
    """Load .npz files and return a list of dicts with extracted arrays."""
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

        trajectories.append({
            "index": i,
            "filename": filename,
            "time_steps": np.arange(obs_real.shape[0]),
            "action_x": actions[:, 0],
            "action_y": actions[:, 1],
            # Velocities at indices [2, 3]
            "vel_real_x": obs_real[:, 2],
            "vel_real_y": obs_real[:, 3],
            "vel_sim_x": obs_sim[:, 2],
            "vel_sim_y": obs_sim[:, 3],
            # Own peg positions at indices [0, 1]
            "pos_real_x": obs_real[:, 0],
            "pos_real_y": obs_real[:, 1],
            "pos_sim_x": obs_sim[:, 0],
            "pos_sim_y": obs_sim[:, 1],
        })

    return trajectories


def calculate_metrics(trajectories):
    """Calculate MSE and RMSE for velocities. Prints and returns a dict."""
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


def plot_velocity_comparison(trajectories):
    """Output 1: Commanded vs real vs sim velocities (test1.py style)."""
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Sim-to-Real Actuator Model Validation", fontsize=18, weight="bold")
    ax_flat = axes.flatten()

    for t in trajectories:
        ax = ax_flat[t["index"]]
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

    handles, labels = ax_flat[0].get_legend_handles_labels()
    ax_flat[-1].legend(handles, labels, loc="center", fontsize=12)
    ax_flat[-1].axis("off")

    _save_or_show(fig, "sim2real_evaluation_01_velocity_comparison.png")
    plt.close(fig)


def plot_performance_metrics(metrics):
    """Output 2: Bar charts of RMSE per trajectory for X and Y velocities."""
    labels = [f"Traj {i+1}" for i in sorted(metrics)]
    rmse_x_vals = [metrics[i]["rmse_x"] for i in sorted(metrics)]
    rmse_y_vals = [metrics[i]["rmse_y"] for i in sorted(metrics)]
    mean_rmse_x = np.mean(rmse_x_vals)
    mean_rmse_y = np.mean(rmse_y_vals)

    x = np.arange(len(labels))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Actuator Model Performance Metrics (RMSE)", fontsize=16, weight="bold")

    bars1 = ax1.bar(x, rmse_x_vals, width, color="tab:blue", alpha=0.8)
    ax1.axhline(mean_rmse_x, color="tab:red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_rmse_x:.4f}")
    ax1.set_title("RMSE — Velocity X")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("RMSE (m/s)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar, val in zip(bars1, rmse_x_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars2 = ax2.bar(x, rmse_y_vals, width, color="tab:orange", alpha=0.8)
    ax2.axhline(mean_rmse_y, color="tab:red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_rmse_y:.4f}")
    ax2.set_title("RMSE — Velocity Y")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("RMSE (m/s)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar, val in zip(bars2, rmse_y_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    print(f"\n  Mean RMSE X: {mean_rmse_x:.6f}   Mean RMSE Y: {mean_rmse_y:.6f}")

    _save_or_show(fig, "sim2real_evaluation_02_performance_metrics.png")
    plt.close(fig)


def plot_velocity_deviation(trajectories):
    """Output 3: Velocity deviation (sim - real) over time."""
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Sim-to-Real Velocity Deviation", fontsize=18, weight="bold")
    ax_flat = axes.flatten()

    for t in trajectories:
        ax = ax_flat[t["index"]]
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

    handles, labels = ax_flat[0].get_legend_handles_labels()
    ax_flat[-1].legend(handles, labels, loc="center", fontsize=12)
    ax_flat[-1].axis("off")

    _save_or_show(fig, "sim2real_evaluation_03_velocity_deviation.png")
    plt.close(fig)


def plot_peg_position(trajectories):
    """Output 4: Peg position sim vs real (top) and position deviation (bottom)."""
    # 6 rows = 3 rows for position comparison + 3 rows for position deviation
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 24), constrained_layout=True)
    fig.suptitle("Sim-to-Real Peg Position", fontsize=18, weight="bold")

    top_axes = axes[:3].flatten()  # rows 0-2: position comparison
    bot_axes = axes[3:].flatten()  # rows 3-5: position deviation

    # Section labels via text on first subplot of each section
    top_axes[0].set_title("Sim-to-Real Peg Position Comparison — Trajectory 1", fontsize=11)
    bot_axes[0].set_title("Peg Position Deviation — Trajectory 1", fontsize=11)

    for t in trajectories:
        idx = t["index"]
        ts = t["time_steps"]

        # --- Position comparison ---
        ax_top = top_axes[idx]
        ax_top.plot(ts, t["pos_real_x"], label="Real Pos X", color="tab:blue", linestyle="-", linewidth=2)
        ax_top.plot(ts, t["pos_sim_x"], label="Sim Pos X", color="tab:cyan", linestyle="--", linewidth=2)
        ax_top.plot(ts, t["pos_real_y"], label="Real Pos Y", color="tab:orange", linestyle="-", linewidth=2)
        ax_top.plot(ts, t["pos_sim_y"], label="Sim Pos Y", color="tab:red", linestyle="--", linewidth=2)
        if idx != 0:
            ax_top.set_title(f"Trajectory {idx+1}", fontsize=11)
        ax_top.set_xlabel("Time Steps")
        ax_top.set_ylabel("Position (m)")
        ax_top.grid(True, linestyle=":", alpha=0.6)

        # --- Position deviation ---
        ax_bot = bot_axes[idx]
        dev_pos_x = t["pos_sim_x"] - t["pos_real_x"]
        dev_pos_y = t["pos_sim_y"] - t["pos_real_y"]
        ax_bot.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax_bot.plot(ts, dev_pos_x, label="Dev Pos X (sim−real)", color="tab:blue", linewidth=1.5)
        ax_bot.plot(ts, dev_pos_y, label="Dev Pos Y (sim−real)", color="tab:orange", linewidth=1.5)
        if idx != 0:
            ax_bot.set_title(f"Trajectory {idx+1}", fontsize=11)
        ax_bot.set_xlabel("Time Steps")
        ax_bot.set_ylabel("Position Error (sim − real) (m)")
        ax_bot.grid(True, linestyle=":", alpha=0.6)

    # Legends in the last (empty) subplot of each section
    handles_top, labels_top = top_axes[0].get_legend_handles_labels()
    top_axes[-1].legend(handles_top, labels_top, loc="center", fontsize=12)
    top_axes[-1].axis("off")

    handles_bot, labels_bot = bot_axes[0].get_legend_handles_labels()
    bot_axes[-1].legend(handles_bot, labels_bot, loc="center", fontsize=12)
    bot_axes[-1].axis("off")

    _save_or_show(fig, "sim2real_evaluation_04_peg_position.png")
    plt.close(fig)


if __name__ == "__main__":
    trajectories = load_trajectories(FILENAMES)
    metrics = calculate_metrics(trajectories)
    plot_velocity_comparison(trajectories)
    plot_performance_metrics(metrics)
    plot_velocity_deviation(trajectories)
    plot_peg_position(trajectories)

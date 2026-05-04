"""Visualize real-robot trajectories used for actuator-model training.

Loops over every per-run .npz file under <data-dir>/<split>/ and writes one PNG
per trajectory containing three panels: XY position, velocity command vs time,
and position vs time. Visual style mirrors sim2real_evaluation.py.

The aggregated training files (data_odrive_*.npz) at the split root are
ignored — those are pre-windowed datasets, not raw trajectories.

Example:
    /workspace/isaaclab/_isaac_sim/python.sh visualize_dataset_trajectories.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REQUIRED_KEYS = ("player_pos", "player_vel", "player_actions")


def find_trajectory_files(split_dir):
    return sorted(p for p in split_dir.rglob("*.npz") if not p.name.startswith("data_odrive_"))


def plot_trajectory(npz_path, interval):
    data = np.load(npz_path)
    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        print(f"Skipping {npz_path.name}: missing keys {missing}")
        return False

    pos = data["player_pos"]
    vel = data["player_vel"]
    cmd = data["player_actions"]
    t = np.arange(pos.shape[0]) * interval

    fig, (ax_xy, ax_vel, ax_pos) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(npz_path.stem, fontsize=14, weight="bold")

    ax_xy.plot(pos[:, 0], pos[:, 1], color="tab:blue", linewidth=1.5)
    ax_xy.scatter(pos[0, 0], pos[0, 1], color="tab:green", s=60, zorder=5, label="Start")
    ax_xy.scatter(pos[-1, 0], pos[-1, 1], color="tab:red", s=60, zorder=5, label="End")
    ax_xy.set_title("XY Position", fontsize=12)
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, linestyle=":", alpha=0.6)
    ax_xy.legend(loc="best", fontsize=9)

    ax_vel.step(t, cmd[:, 0], label="Cmd Vel X", color="tab:blue", linestyle="--", linewidth=1, where="post")
    ax_vel.step(t, cmd[:, 1], label="Cmd Vel Y", color="tab:orange", linestyle="--", linewidth=1, where="post")
    ax_vel.plot(t, vel[:, 0], label="Vel X", color="tab:blue", linestyle="-", linewidth=2)
    ax_vel.plot(t, vel[:, 1], label="Vel Y", color="tab:orange", linestyle="-", linewidth=2)
    ax_vel.set_title("Velocity Command vs Time", fontsize=12)
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.grid(True, linestyle=":", alpha=0.6)
    ax_vel.legend(loc="best", fontsize=9)

    ax_pos.plot(t, pos[:, 0], label="Pos X", color="tab:blue", linestyle="-", linewidth=2)
    ax_pos.plot(t, pos[:, 1], label="Pos Y", color="tab:orange", linestyle="-", linewidth=2)
    ax_pos.set_title("Position vs Time", fontsize=12)
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.grid(True, linestyle=":", alpha=0.6)
    ax_pos.legend(loc="best", fontsize=9)

    output_path = npz_path.with_name(f"{npz_path.stem}_visualization.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/klask_rl/logs/actuator_model/data/train_traj"),
        help="Root directory containing per-split trajectory subdirectories.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.02,
        help="Sampling interval in seconds (default: 0.02 = 50 Hz, matches create_dataset.py).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Subdirectory names under --data-dir to process.",
    )
    args = parser.parse_args()

    total = 0
    for split in args.splits:
        split_dir = args.data_dir / split
        if not split_dir.is_dir():
            print(f"Warning: split directory not found, skipping: {split_dir}")
            continue

        files = find_trajectory_files(split_dir)
        print(f"[{split}] Found {len(files)} trajectory file(s) under {split_dir}")
        saved = 0
        for npz_path in files:
            if plot_trajectory(npz_path, args.interval):
                saved += 1
        print(f"[{split}] Saved {saved} visualization(s)")
        total += saved

    print(f"\nSaved {total} visualization(s) across splits: {', '.join(args.splits)}")


if __name__ == "__main__":
    main()

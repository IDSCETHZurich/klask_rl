"""Visualize real-robot trajectories used for actuator-model training.

Loops over every per-run .npz file under <data-dir>/<split>/ and writes one PNG
per trajectory containing three panels: XY position, velocity command vs time,
and position vs time. Visual style mirrors sim2real_evaluation.py.

The aggregated training files (data_odrive_*.npz) at the split root are
ignored — those are pre-windowed datasets, not raw trajectories.

Pass `--target <file_or_dir>` to analyse a single .npz file or a single
trajectory directory instead of the full splits.

Pass `--measure` to additionally run settling-time analysis on line-shaped
trajectories (vel_profile_x_line_* / vel_profile_y_line_*). For each clean
direction-change cycle (partial first/last plateaus are cropped) the script
measures:
  t_decel — time from cmd flip until the peg has stopped (|v| <= ZERO_FRAC * |v_target|)
  t_accel — time from "stopped" until the peg reaches the new commanded velocity
            (sign matches AND |v| >= REACH_FRAC * |v_target|)
The per-trajectory mean +/- std is printed, a <stem>_settling.png verification
plot is written next to the trajectory, and a settling_summary.csv is written
at the data root.

Note on frames: file names are in image frame, but the data is in ego frame
(axes swapped). So `*_x_line_*` files command on cmd[:,1] (ego-y) and
`*_y_line_*` files command on cmd[:,0] (ego-x).

Example:
    /workspace/isaaclab/_isaac_sim/python.sh visualize_dataset_trajectories.py
    /workspace/isaaclab/_isaac_sim/python.sh visualize_dataset_trajectories.py \\
        --measure --target /path/to/vel_profile_y_line_..._run9
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

REQUIRED_KEYS = ("player_pos", "player_vel", "player_actions")

ZERO_FRAC = 0.05
REACH_FRAC = 0.95


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


def detect_active_axis(npz_path):
    """Return the ego-frame axis index (0=x, 1=y) for line-shaped trajectories.

    File names are in image frame, data is in ego frame, so the axes are swapped:
        *_x_line_* (image) -> ego-y line -> axis 1
        *_y_line_* (image) -> ego-x line -> axis 0
    Returns None for non-line trajectories.
    """
    stem = npz_path.stem
    if "x_line" in stem:
        return 1
    if "y_line" in stem:
        return 0
    return None


def find_transitions(cmd_axis, atol=1e-4):
    diffs = np.abs(np.diff(cmd_axis))
    transitions = np.where(diffs > atol)[0]
    plateau_starts = np.concatenate(([0], transitions + 1))
    plateau_lengths = np.diff(np.concatenate((plateau_starts, [len(cmd_axis)])))
    return transitions, plateau_starts, plateau_lengths


def clean_transition_mask(plateau_lengths, min_len):
    n_trans = len(plateau_lengths) - 1
    mask = np.zeros(n_trans, dtype=bool)
    for i in range(n_trans):
        mask[i] = plateau_lengths[i] >= min_len and plateau_lengths[i + 1] >= min_len
    return mask


def measure_transition(vel_axis, cmd_axis, k, dt, search_window):
    v_old = float(cmd_axis[k])
    v_new = float(cmd_axis[k + 1])
    v_target = abs(v_new)
    zero_thresh = ZERO_FRAC * v_target
    reach_thresh = REACH_FRAC * v_target
    sign_old = np.sign(v_old)
    sign_new = np.sign(v_new)
    search_end = min(k + 1 + search_window, len(vel_axis))

    i_zero = None
    for j in range(k + 1, search_end):
        if abs(vel_axis[j]) <= zero_thresh or np.sign(vel_axis[j]) != sign_old:
            i_zero = j
            break

    i_reach = None
    if i_zero is not None:
        for j in range(i_zero, search_end):
            if np.sign(vel_axis[j]) == sign_new and abs(vel_axis[j]) >= reach_thresh:
                i_reach = j
                break

    valid = i_zero is not None and i_reach is not None
    # Cmd flip is effective at sample k+1 (where cmd_axis[k+1] != cmd_axis[k]),
    # which is also where step(..., where="post") draws the visual step.
    return {
        "k": int(k),
        "i_zero": int(i_zero) if i_zero is not None else -1,
        "i_reach": int(i_reach) if i_reach is not None else -1,
        "t_decel": (i_zero - (k + 1)) * dt if i_zero is not None else float("nan"),
        "t_accel": (i_reach - i_zero) * dt if valid else float("nan"),
        "t_total": (i_reach - (k + 1)) * dt if valid else float("nan"),
        "v_target": v_target,
        "valid": valid,
    }


def analyze_settling(npz_path, interval):
    """Run settling-time analysis and write <stem>_settling.png. Returns list of measurement dicts."""
    axis = detect_active_axis(npz_path)
    if axis is None:
        return []

    data = np.load(npz_path)
    if any(k not in data.files for k in REQUIRED_KEYS):
        return []

    cmd = data["player_actions"]
    vel = data["player_vel"]
    pos = data["player_pos"]
    cmd_axis = cmd[:, axis]
    vel_axis = vel[:, axis]
    pos_axis = pos[:, axis]
    other_axis_ptp = float(cmd[:, 1 - axis].ptp())
    active_ptp = float(cmd_axis.ptp())
    if other_axis_ptp > active_ptp:
        print(
            f"[warn] {npz_path.name}: other axis cmd ptp ({other_axis_ptp:.3f}) > active ({active_ptp:.3f}); skipping"
        )
        return []

    transitions, plateau_starts, plateau_lengths = find_transitions(cmd_axis)
    if len(transitions) < 3:
        print(f"[warn] {npz_path.name}: only {len(transitions)} transitions; skipping settling analysis")
        return []

    median_len = float(np.median(plateau_lengths))
    min_len = max(3, int(0.8 * median_len))
    mask = clean_transition_mask(plateau_lengths, min_len)
    clean_idxs = np.where(mask)[0]
    if len(clean_idxs) < 3:
        print(
            f"[warn] {npz_path.name}: only {len(clean_idxs)} clean cycle(s) (need at least 3 to drop edges); skipping"
        )
        return []
    # Also drop the first and last clean cycle so we only analyse fully-settled
    # cycles bracketed on both sides by other clean cycles.
    clean_idxs = clean_idxs[1:-1]

    search_window = max(int(median_len), 5)

    measurements = []
    for trans_idx in clean_idxs:
        k = int(transitions[trans_idx])
        m = measure_transition(vel_axis, cmd_axis, k, interval, search_window)
        m["trans_idx"] = int(trans_idx)
        measurements.append(m)

    valid = [m for m in measurements if m["valid"]]
    if valid:
        td = np.array([m["t_decel"] for m in valid])
        ta = np.array([m["t_accel"] for m in valid])
        tt = np.array([m["t_total"] for m in valid])
        v_target = valid[0]["v_target"]
        print(
            f"[measure] {npz_path.stem}: axis={axis} v_target={v_target:.3f} "
            f"n={len(valid)}/{len(measurements)} "
            f"t_decel={td.mean()*1000:.1f}+/-{td.std()*1000:.1f} ms "
            f"t_accel={ta.mean()*1000:.1f}+/-{ta.std()*1000:.1f} ms "
            f"t_total={tt.mean()*1000:.1f}+/-{tt.std()*1000:.1f} ms"
        )
    else:
        print(f"[warn] {npz_path.name}: no valid measurements out of {len(measurements)} clean cycles")

    plot_settling(
        npz_path,
        interval,
        axis,
        cmd_axis,
        vel_axis,
        pos_axis,
        transitions,
        plateau_starts,
        plateau_lengths,
        clean_idxs,
        measurements,
    )
    return measurements


def plot_settling(
    npz_path,
    dt,
    axis,
    cmd_axis,
    vel_axis,
    pos_axis,
    transitions,
    plateau_starts,
    plateau_lengths,
    clean_idxs,
    measurements,
):
    n = len(vel_axis)
    t = np.arange(n) * dt
    axis_label = "X" if axis == 0 else "Y"

    first_plateau_idx = int(clean_idxs[0])
    last_plateau_idx = int(clean_idxs[-1]) + 1
    crop_start = int(plateau_starts[first_plateau_idx])
    crop_end = min(int(plateau_starts[last_plateau_idx] + plateau_lengths[last_plateau_idx]), n)

    rep_i = int(clean_idxs[0])
    rep_meas = next((m for m in measurements if m["trans_idx"] == rep_i and m["valid"]), None)
    if rep_meas is None:
        rep_meas = next((m for m in measurements if m["valid"]), None)
        if rep_meas is not None:
            rep_i = rep_meas["trans_idx"]
    rep_zoom_start = int(plateau_starts[rep_i])
    rep_end_idx = min(rep_i + 2, len(plateau_starts) - 1)
    rep_zoom_end = min(int(plateau_starts[rep_end_idx] + plateau_lengths[rep_end_idx]), n)

    fig, (ax_top, ax_bot, ax_pos) = plt.subplots(nrows=3, ncols=1, figsize=(14, 11), constrained_layout=True)
    n_valid = sum(1 for m in measurements if m["valid"])
    fig.suptitle(
        f"{npz_path.stem} - settling analysis (active ego-{axis_label.lower()}, n={n_valid}/{len(measurements)} valid)",
        fontsize=13,
        weight="bold",
    )

    sl = slice(crop_start, crop_end)
    ax_top.step(
        t[sl], cmd_axis[sl], where="post", color="tab:blue", linestyle="--", linewidth=1, label=f"Cmd Vel {axis_label}"
    )
    ax_top.plot(t[sl], vel_axis[sl], color="tab:blue", linewidth=2, label=f"Vel {axis_label}")
    ax_top.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    decel_label_done = False
    accel_label_done = False
    for m in measurements:
        if not m["valid"]:
            continue
        ax_top.axvspan(
            t[m["k"] + 1], t[m["i_zero"]], color="tab:orange", alpha=0.3, label=None if decel_label_done else "t_decel"
        )
        ax_top.axvspan(
            t[m["i_zero"]], t[m["i_reach"]], color="tab:green", alpha=0.3, label=None if accel_label_done else "t_accel"
        )
        decel_label_done = True
        accel_label_done = True
    ax_top.set_xlim(t[crop_start], t[crop_end - 1])
    ax_top.set_title("Cropped clean cycles", fontsize=11)
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel(f"Vel {axis_label} (m/s)")
    ax_top.grid(True, linestyle=":", alpha=0.6)
    ax_top.legend(loc="best", fontsize=9)

    if rep_meas is not None:
        sl2 = slice(rep_zoom_start, rep_zoom_end)
        v_target = rep_meas["v_target"]
        zero_thresh = ZERO_FRAC * v_target
        reach_thresh = REACH_FRAC * v_target
        ax_bot.step(
            t[sl2],
            cmd_axis[sl2],
            where="post",
            color="tab:blue",
            linestyle="--",
            linewidth=1,
            label=f"Cmd Vel {axis_label}",
        )
        ax_bot.plot(t[sl2], vel_axis[sl2], color="tab:blue", linewidth=2, label=f"Vel {axis_label}")
        ax_bot.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax_bot.axhline(zero_thresh, color="tab:gray", linestyle=":", linewidth=1, label=f"+/-zero ({zero_thresh:.3f})")
        ax_bot.axhline(-zero_thresh, color="tab:gray", linestyle=":", linewidth=1)
        ax_bot.axhline(
            reach_thresh, color="tab:purple", linestyle=":", linewidth=1, label=f"+/-reach ({reach_thresh:.3f})"
        )
        ax_bot.axhline(-reach_thresh, color="tab:purple", linestyle=":", linewidth=1)
        ax_bot.axvline(t[rep_meas["k"] + 1], color="tab:red", linestyle="--", linewidth=1, label="cmd flip")
        ax_bot.axvline(
            t[rep_meas["i_zero"]],
            color="tab:orange",
            linestyle="--",
            linewidth=1,
            label=f"stopped (t_decel={rep_meas['t_decel']*1000:.1f} ms)",
        )
        ax_bot.axvline(
            t[rep_meas["i_reach"]],
            color="tab:green",
            linestyle="--",
            linewidth=1,
            label=f"reached (t_accel={rep_meas['t_accel']*1000:.1f} ms)",
        )
        ax_bot.set_title(
            f"Representative cycle (transition #{rep_meas['trans_idx']}) - t_total={rep_meas['t_total']*1000:.1f} ms",
            fontsize=11,
        )
        ax_bot.set_xlabel("Time (s)")
        ax_bot.set_ylabel(f"Vel {axis_label} (m/s)")
        ax_bot.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_bot.grid(True, which="major", linestyle=":", alpha=0.6)
        ax_bot.grid(True, which="minor", axis="x", linestyle=":", alpha=0.3)
        ax_bot.legend(loc="best", fontsize=8)

        ax_pos.plot(t[sl2], pos_axis[sl2], color="tab:blue", linewidth=2, label=f"Pos {axis_label}")
        ax_pos.axvline(t[rep_meas["k"] + 1], color="tab:red", linestyle="--", linewidth=1, label="cmd flip")
        ax_pos.axvline(t[rep_meas["i_zero"]], color="tab:orange", linestyle="--", linewidth=1, label="stopped")
        ax_pos.axvline(t[rep_meas["i_reach"]], color="tab:green", linestyle="--", linewidth=1, label="reached")
        ax_pos.set_title("Position over the same cycle", fontsize=11)
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel(f"Pos {axis_label} (m)")
        ax_pos.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_pos.grid(True, which="major", linestyle=":", alpha=0.6)
        ax_pos.grid(True, which="minor", axis="x", linestyle=":", alpha=0.3)
        ax_pos.legend(loc="best", fontsize=8)
    else:
        ax_bot.text(0.5, 0.5, "no valid measurement", ha="center", va="center", transform=ax_bot.transAxes)
        ax_pos.text(0.5, 0.5, "no valid measurement", ha="center", va="center", transform=ax_pos.transAxes)

    out = npz_path.with_name(f"{npz_path.stem}_settling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gather_targets(args):
    """Yield (label, npz_path) tuples for all trajectories to process."""
    if args.target is not None:
        target = args.target
        if not target.exists():
            print(f"Error: --target not found: {target}")
            return []
        if target.is_file():
            if target.suffix != ".npz" or target.name.startswith("data_odrive_"):
                print(f"Error: --target must be a per-run .npz file, got {target.name}")
                return []
            label = target.parent.name or "."
            print(f"[target] Single file under label '{label}': {target.name}")
            return [(label, target)]
        if target.is_dir():
            files = find_trajectory_files(target)
            label = target.name
            print(f"[target] {len(files)} trajectory file(s) under {target}")
            return [(label, p) for p in files]
        print(f"Error: --target is neither a file nor a directory: {target}")
        return []

    out = []
    for split in args.splits:
        split_dir = args.data_dir / split
        if not split_dir.is_dir():
            print(f"Warning: split directory not found, skipping: {split_dir}")
            continue
        files = find_trajectory_files(split_dir)
        print(f"[{split}] Found {len(files)} trajectory file(s) under {split_dir}")
        out.extend((split, p) for p in files)
    return out


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
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Path to a single .npz file or a single trajectory directory; overrides --data-dir/--splits.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Also run settling-time analysis on x_line / y_line trajectories.",
    )
    args = parser.parse_args()

    targets = gather_targets(args)
    saved = 0
    all_measurements = []
    for label, npz_path in targets:
        if plot_trajectory(npz_path, args.interval):
            saved += 1
        if args.measure:
            ms = analyze_settling(npz_path, args.interval)
            axis = detect_active_axis(npz_path)
            for m in ms:
                all_measurements.append((label, npz_path.stem, axis, m))

    print(f"\nSaved {saved} visualization(s).")

    if args.measure and all_measurements:
        if args.target is not None and args.target.is_dir():
            csv_dir = args.target
        elif args.target is not None and args.target.is_file():
            csv_dir = args.target.parent
        else:
            csv_dir = args.data_dir
        csv_path = csv_dir / "settling_summary.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "split",
                "stem",
                "axis",
                "v_target",
                "k",
                "i_zero",
                "i_reach",
                "t_decel_s",
                "t_accel_s",
                "t_total_s",
                "valid",
            ])
            for label, stem, axis, m in all_measurements:
                w.writerow([
                    label,
                    stem,
                    axis,
                    m["v_target"],
                    m["k"],
                    m["i_zero"],
                    m["i_reach"],
                    m["t_decel"],
                    m["t_accel"],
                    m["t_total"],
                    int(m["valid"]),
                ])
        print(f"Wrote settling CSV: {csv_path}")

        for pattern in ("x_line", "y_line"):
            valid = [m for (_, stem, _, m) in all_measurements if pattern in stem and m["valid"]]
            if not valid:
                continue
            td = np.array([m["t_decel"] for m in valid])
            ta = np.array([m["t_accel"] for m in valid])
            tt = np.array([m["t_total"] for m in valid])
            print(
                f"[aggregate {pattern}] n={len(valid)} "
                f"t_decel={td.mean()*1000:.1f}+/-{td.std()*1000:.1f} ms "
                f"t_accel={ta.mean()*1000:.1f}+/-{ta.std()*1000:.1f} ms "
                f"t_total={tt.mean()*1000:.1f}+/-{tt.std()*1000:.1f} ms"
            )


if __name__ == "__main__":
    main()

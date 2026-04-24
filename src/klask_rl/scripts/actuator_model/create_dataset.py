#!/usr/bin/env python3
"""Create actuator model training dataset from ROS 2 bag (.db3) trajectory files.

Reads /board_state and /cmd_vel/left_player topics from .db3 bags,
builds windowed input-output pairs, and saves as .npz for learn_actuator.py.
"""

import argparse
import os
import sqlite3
import struct
from pathlib import Path

import numpy as np

_BOARD_WIDTH_M = 0.32
_BOARD_HEIGHT_M = 0.42

# ---------------------------------------------------------------------------
# Image-frame → centered sim-portrait coordinate transform
# ---------------------------------------------------------------------------
# Image frame (as published on /board_state): landscape, origin top-left,
# x → right along the long axis, y → down along the short axis.
# Sim frame: portrait, origin at board center. 90° CCW rotation + translation.

_CENTER_X_IMG = _BOARD_HEIGHT_M / 2.0  # long axis (raw_x)
_CENTER_Y_IMG = _BOARD_WIDTH_M / 2.0  # short axis (raw_y)


def _transform_object_state(px, py, vx, vy):
    """Image frame → centered sim portrait.

    Positions: 90° CCW rotation + translation to board center.
    Velocities: rotation only (no translation).
    """
    sim_px = _CENTER_Y_IMG - py
    sim_py = px - _CENTER_X_IMG
    sim_vx = -vy
    sim_vy = vx
    return sim_px, sim_py, sim_vx, sim_vy


def _transform_command(linear_x, linear_y):
    """Rotation-only image → sim."""
    return -linear_y, linear_x


# ---------------------------------------------------------------------------
# Windowing helpers (copied from old_mcap_dataloader.py — cannot import
# because rosbag2_py/rclpy are not installed)
# ---------------------------------------------------------------------------


def find_nearest_indices(time_array, target_time):
    idx = np.searchsorted(time_array, target_time)
    if idx == 0:
        return 0, 0, 0
    if idx >= len(time_array):
        return len(time_array) - 1, len(time_array) - 1, len(time_array) - 1
    before = idx - 1
    after = idx
    nearest = before if (target_time - time_array[before]) < (time_array[after] - target_time) else after
    return before, after, nearest


def interp_linear(values_2d, times_1d, target_times_1d):
    """Per-column linear interpolation onto ``target_times_1d``.

    values_2d: (N, D), times_1d: (N,), target_times_1d: (T,) → (T, D) float32.
    Out-of-range targets clamp to boundary samples (np.interp semantics).
    """
    values_2d = np.asarray(values_2d, dtype=np.float32)
    t_src = np.asarray(times_1d, dtype=np.float64)
    t_dst = np.atleast_1d(np.asarray(target_times_1d, dtype=np.float64))
    out = np.empty((t_dst.shape[0], values_2d.shape[1]), dtype=np.float32)
    for d in range(values_2d.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, values_2d[:, d])
    return out


def build_dataset_interpolated(
    states_list,
    state_times,
    commands_list,
    cmd_times,
    num_history_steps,
    interval_size,
    delay,
    horizon=1,
    include_states=False,
):
    X_commands = []
    X_states = []
    Y = []
    Y_prev = []
    commands = []
    delay = int(delay * 1e9)
    interval_size = int(interval_size * 1e9)

    states_arr = np.asarray(states_list, dtype=np.float32)
    state_times_arr = np.asarray(state_times)

    prev_state = None

    for i, (t, state) in enumerate(zip(state_times, states_arr)):
        if t + horizon * interval_size > state_times[-1]:
            continue
        if prev_state is None:
            prev_state = state
        hist_vecs = []
        if t - delay - num_history_steps * interval_size < cmd_times[0]:
            continue
        for j in range(num_history_steps):
            target_time = t - delay - j * interval_size
            before, after, _ = find_nearest_indices(cmd_times, target_time)
            hist_vecs.append(commands_list[before])

        if include_states:
            hist_targets = np.array(
                [t - delay - j * interval_size for j in range(1, num_history_steps)],
                dtype=np.int64,
            )
            state_hist_interp = interp_linear(states_arr, state_times_arr, hist_targets)
            state_hist_vecs = [row[2:] for row in state_hist_interp]

        hist_vec = np.hstack(hist_vecs)
        X_commands.append(hist_vec)
        if include_states:
            state_hist_vec = np.hstack(state_hist_vecs)
            X_states.append(state_hist_vec)

        horizon_targets = np.array([t + j * interval_size for j in range(horizon)], dtype=np.int64)
        horizon_states = interp_linear(states_arr, state_times_arr, horizon_targets)

        command_vecs = []
        y_vecs = []
        y_prev_vecs = [prev_state[2:]]
        for j in range(horizon):
            target_time = horizon_targets[j]
            next_state = horizon_states[j, 2:]
            y_vecs.append(next_state)
            if j < horizon - 1:
                y_prev_vecs.append(next_state)
            before, after, _ = find_nearest_indices(cmd_times, target_time)
            command_vecs.append(commands_list[before])

        Y.append(np.vstack(y_vecs))
        Y_prev.append(np.vstack(y_prev_vecs))
        commands.append(np.vstack(command_vecs))
        prev_state = state

    X_commands = np.array(X_commands, dtype=np.float32)
    if include_states:
        X_states = np.array(X_states, dtype=np.float32)
    else:
        X_states = None
    Y = np.array(Y, dtype=np.float32)
    Y_prev = np.array(Y_prev, dtype=np.float32)
    commands = np.array(commands, dtype=np.float32)

    return X_commands, X_states, Y, Y_prev, commands


# ---------------------------------------------------------------------------
# DB3 reader
# ---------------------------------------------------------------------------


def load_data_from_db3(bag_dir, state_topic="/board_state", command_topic="/cmd_vel/left_player"):
    """Load state and command data from a ROS 2 bag (.db3) directory.

    Applies the image-frame → centered sim-portrait transform (90° CCW
    rotation + translation for positions, rotation-only for velocities and
    commands) to every field.

    Returns:
        (state_times,       # (N,) int64 ns
         player_states,     # (N, 4) float32 [px, py, vx, vy] sim
         opponent_states,   # (N, 4) float32
         ball_states,       # (N, 4) float32
         cmd_times,         # (M,) int64 ns
         player_commands)   # (M, 2) float32 [cx, cy] sim
    """
    empty = (None, None, None, None, None, None)
    bag_path = Path(bag_dir)
    db3_files = list(bag_path.glob("*.db3"))
    if not db3_files:
        print(f"  Warning: No .db3 file in {bag_dir}")
        return empty
    db3_file = db3_files[0]

    conn = sqlite3.connect(str(db3_file))

    topics = conn.execute("SELECT id, name FROM topics").fetchall()
    state_id = None
    cmd_id = None
    for tid, tname in topics:
        if tname == state_topic:
            state_id = tid
        elif tname == command_topic:
            cmd_id = tid

    if state_id is None or cmd_id is None:
        print(f"  Warning: Missing topics in {bag_dir} (state_id={state_id}, cmd_id={cmd_id})")
        conn.close()
        return empty

    # Read /board_state messages.
    # CDR layout: 4-byte encap header + 27 float64 values (216 bytes) = 220 bytes.
    # Field offsets within the 27-double block (empirically verified):
    #   vals[2:8]   = ball  ObjectState (pos.xyz, vel.xyz)
    #   vals[8:14]  = left_peg  → player
    #   vals[14:20] = right_peg → opponent
    state_records = []
    for ts, data in conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (state_id,)
    ):
        vals = struct.unpack_from("<27d", data, 4)
        player = _transform_object_state(vals[8], vals[9], vals[11], vals[12])
        opponent = _transform_object_state(vals[14], vals[15], vals[17], vals[18])
        ball = _transform_object_state(vals[2], vals[3], vals[5], vals[6])
        state_records.append((ts, player, opponent, ball))

    # Read /cmd_vel messages (Twist: linear xyz + angular xyz = 6 doubles).
    cmd_records = []
    for ts, data in conn.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (cmd_id,)):
        vals = struct.unpack_from("<6d", data, 4)
        cx, cy = _transform_command(vals[0], vals[1])
        cmd_records.append((ts, cx, cy))

    conn.close()

    if len(state_records) == 0 or len(cmd_records) == 0:
        print(f"  Warning: Empty data in {bag_dir}")
        return empty

    state_times = np.array([r[0] for r in state_records], dtype=np.int64)
    player_states = np.array([r[1] for r in state_records], dtype=np.float32)
    opponent_states = np.array([r[2] for r in state_records], dtype=np.float32)
    ball_states = np.array([r[3] for r in state_records], dtype=np.float32)
    cmd_times = np.array([r[0] for r in cmd_records], dtype=np.int64)
    player_commands = np.array([(r[1], r[2]) for r in cmd_records], dtype=np.float32)

    return state_times, player_states, opponent_states, ball_states, cmd_times, player_commands


# ---------------------------------------------------------------------------
# Per-trajectory replay npz (consumed by playback_actions.py)
# ---------------------------------------------------------------------------


def resample_and_save_replay_npz(
    out_path,
    state_times,
    player_states,
    opponent_states,
    ball_states,
    cmd_times,
    player_commands,
    interval_s,
):
    """Resample trajectory onto a fixed-rate grid and save as .npz for playback."""
    start_ns = int(max(state_times[0], cmd_times[0]))
    end_ns = int(min(state_times[-1], cmd_times[-1]))
    step_ns = int(interval_s * 1e9)
    grid = np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)

    p = interp_linear(player_states, state_times, grid)
    o = interp_linear(opponent_states, state_times, grid)
    b = interp_linear(ball_states, state_times, grid)
    # Commands are piecewise-constant step inputs → zero-order hold.
    idx = np.clip(np.searchsorted(cmd_times, grid, side="right") - 1, 0, len(cmd_times) - 1)
    a = player_commands[idx]

    np.savez(
        out_path,
        player_pos=p[:, :2],
        player_vel=p[:, 2:],
        opponent_pos=o[:, :2],
        opponent_vel=o[:, 2:],
        ball_pos=b[:, :2],
        ball_vel=b[:, 2:],
        player_actions=a,
    )
    return grid.shape[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_split(data_dir: Path, output_path: str, args, include_states: bool):
    """Load all trajectories under data_dir, build windowed dataset, save to output_path."""
    bag_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(bag_dirs)} trajectory directories in {data_dir}")

    extracted = []
    for bag_dir in bag_dirs:
        print(f"Loading {bag_dir.name}...")
        result = load_data_from_db3(str(bag_dir))
        if result[0] is None:
            continue
        state_times, player_states, opponent_states, ball_states, cmd_times, player_commands = result

        replay_path = bag_dir / f"{bag_dir.name}.npz"
        n_steps = resample_and_save_replay_npz(
            replay_path,
            state_times,
            player_states,
            opponent_states,
            ball_states,
            cmd_times,
            player_commands,
            args.interval,
        )
        print(f"  → wrote replay npz {replay_path.name} ({n_steps} steps)")

        # Feed the windowed pipeline — player_states plays the role of states_list.
        extracted.append((player_states, state_times, player_commands, cmd_times))

    if not extracted:
        print("No valid data found.")
        return

    print(f"\nLoaded {len(extracted)} trajectories successfully.")

    X_commands_all = []
    X_states_all = []
    Y_all = []
    Y_prev_all = []
    commands_all = []

    for states_list, state_times, commands_list, cmd_times in extracted:
        X_commands, X_states, Y, Y_prev, commands = build_dataset_interpolated(
            states_list,
            state_times,
            commands_list,
            cmd_times,
            args.history,
            args.interval,
            args.delay,
            args.horizon,
            include_states,
        )
        if X_commands is not None and len(X_commands) > 0:
            X_commands_all.append(X_commands)
            if X_states is not None:
                X_states_all.append(X_states)
            Y_all.append(Y)
            Y_prev_all.append(Y_prev)
            commands_all.append(commands)
            print(f"  → {X_commands.shape[0]} samples")
        else:
            print("  → 0 samples (skipped)")

    if not X_commands_all:
        print("No samples produced.")
        return

    X_commands_full = np.vstack(X_commands_all)
    Y_full = np.vstack(Y_all)
    Y_prev_full = np.vstack(Y_prev_all)
    commands_full = np.vstack(commands_all)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if include_states:
        X_states_full = np.vstack(X_states_all)
        np.savez(
            output_path,
            X_commands=X_commands_full,
            X_states=X_states_full,
            Y=Y_full,
            Y_prev=Y_prev_full,
            commands=commands_full,
        )
    else:
        np.savez(
            output_path,
            X_commands=X_commands_full,
            Y=Y_full,
            Y_prev=Y_prev_full,
            commands=commands_full,
        )

    print(f"\nSaved {X_commands_full.shape[0]} samples to {output_path}")
    print(f"  X_commands: {X_commands_full.shape}")
    if include_states:
        print(f"  X_states:   {X_states_full.shape}")
    print(f"  Y:          {Y_full.shape}")
    print(f"  Y_prev:     {Y_prev_full.shape}")
    print(f"  commands:   {commands_full.shape}")

    input_dim = X_commands_full.shape[1]
    if include_states:
        input_dim += X_states_full.shape[1]
    print(f"  Network input dim: {input_dim}")

    all_vels = Y_full.reshape(-1, 2)
    all_cmds = commands_full.reshape(-1, 2)
    print(f"\n  Velocity range: [{all_vels.min():.4f}, {all_vels.max():.4f}] m/s")
    print(f"  Command range:  [{all_cmds.min():.4f}, {all_cmds.max():.4f}] m/s")


def build_output_path(args, include_states: bool, output_dir: Path, split: str | None = None) -> str:
    states_suffix = "_with_states" if include_states else ""
    split_suffix = f"_{split}" if split else ""
    filename = (
        f"data_odrive_new_estimator_history_{args.history}"
        f"_interval_{args.interval}_delay_{args.delay}"
        f"_horizon{args.horizon}{states_suffix}{split_suffix}.npz"
    )
    return str(output_dir / filename)


def main():
    parser = argparse.ArgumentParser(description="Create actuator model dataset from .db3 trajectory bags")
    parser.add_argument(
        "--data_dir",
        default=str(Path(__file__).resolve().parents[2] / "logs" / "actuator_model" / "data" / "train_traj"),
        help=(
            "Path to directory containing trajectory subdirectories (optionally with train/ and validation/ subfolders)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz file path (auto-generated if omitted; ignored when train/validation split is detected)",
    )
    parser.add_argument("--history", type=int, default=10, help="Number of past command/state steps (default: 10)")
    parser.add_argument(
        "--interval", type=float, default=0.02, help="Time between history steps in seconds (default: 0.02)"
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Command-to-state delay in seconds (default: 0.0)")
    parser.add_argument("--horizon", type=int, default=3, help="Future prediction steps (default: 3)")
    parser.add_argument("--no-states", action="store_true", help="Omit velocity state history from input")
    args = parser.parse_args()

    include_states = not args.no_states
    data_dir = Path(args.data_dir)

    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"

    if train_dir.is_dir() and val_dir.is_dir():
        if args.output:
            print("Note: --output is ignored when train/ and validation/ subfolders are detected.")
        for split, split_dir in (("train", train_dir), ("validation", val_dir)):
            print(f"\n=== Processing {split} split ({split_dir}) ===")
            output_path = build_output_path(args, include_states, split_dir, split=split)
            process_split(split_dir, output_path, args, include_states)
    else:
        output_path = args.output if args.output else build_output_path(args, include_states, data_dir)
        process_split(data_dir, output_path, args, include_states)


if __name__ == "__main__":
    main()

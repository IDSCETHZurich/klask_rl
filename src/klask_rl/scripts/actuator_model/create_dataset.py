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
    # Convert delay and interval from seconds to nanoseconds
    delay = int(delay * 1e9)
    interval_size = int(interval_size * 1e9)

    prev_state = None

    for i, (t, state) in enumerate(zip(state_times, states_list)):
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
            state_hist_vecs = []
            for j in range(1, num_history_steps):
                target_time = t - delay - j * interval_size
                before, after, _ = find_nearest_indices(state_times, target_time)
                if state_times[after] > t - interval_size:
                    if before > 0 and state_times[before] != state_times[before - 1]:
                        slope = (states_list[before] - states_list[before - 1]) / (
                            state_times[before] - state_times[before - 1]
                        )
                    else:
                        slope = np.zeros_like(states_list[before])
                elif state_times[after] != state_times[before]:
                    slope = (states_list[after] - states_list[before]) / (state_times[after] - state_times[before])
                else:
                    slope = np.zeros_like(states_list[before])
                state_hist_vecs.append((states_list[before] + slope * (target_time - state_times[before]))[2:])

        hist_vec = np.hstack(hist_vecs)
        X_commands.append(hist_vec)
        if include_states:
            state_hist_vec = np.hstack(state_hist_vecs)
            X_states.append(state_hist_vec)

        command_vecs = []
        y_vecs = []
        y_prev_vecs = []
        y_prev_vecs.append(prev_state[2:])
        for j in range(horizon):
            target_time = t + j * interval_size
            before, after, _ = find_nearest_indices(state_times, target_time)
            if state_times[after] != state_times[before]:
                slope = (states_list[after] - states_list[before]) / (state_times[after] - state_times[before])
            else:
                slope = np.zeros_like(states_list[before])
            next_state = (states_list[before] + slope * (target_time - state_times[before]))[2:]
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

    Returns the same interface as old_mcap_dataloader.load_data_from_bag():
        (states_list, state_times, commands_list, cmd_times)

    States are 4D [pos_y, pos_x, vel_y, vel_x] (axes swapped to match sim convention).
    Commands are 2D [cmd_y, cmd_x] (axes swapped to match sim convention).
    No unit conversion — data is already in SI (m, m/s).
    """
    # Find .db3 file
    bag_path = Path(bag_dir)
    db3_files = list(bag_path.glob("*.db3"))
    if not db3_files:
        print(f"  Warning: No .db3 file in {bag_dir}")
        return None, None, None, None
    db3_file = db3_files[0]

    conn = sqlite3.connect(str(db3_file))

    # Look up topic IDs
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
        return None, None, None, None

    # Read board_state messages
    # CDR layout: 4-byte header + 27 float64 values (216 bytes) = 220 bytes total
    # Left player: double[8]=pos_x, double[9]=pos_y, double[11]=vel_x, double[12]=vel_y
    state_records = []
    for ts, data in conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (state_id,)
    ):
        vals = struct.unpack_from("<27d", data, 4)
        # Swap axes: [pos_y, pos_x, vel_y, vel_x] to match sim [y, x] convention
        state_vec = np.array([vals[9], vals[8], vals[12], vals[11]], dtype=np.float32)
        state_records.append((ts, state_vec))

    # Read cmd_vel messages
    # CDR layout: 4-byte header + 6 float64 (Twist: linear xyz + angular xyz) = 52 bytes
    cmd_records = []
    for ts, data in conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (cmd_id,)
    ):
        vals = struct.unpack_from("<6d", data, 4)
        # Swap axes: [cmd_y, cmd_x] to match sim [y, x] convention
        cmd_vec = np.array([vals[1], vals[0]], dtype=np.float32)
        cmd_records.append((ts, cmd_vec))

    conn.close()

    if len(state_records) == 0 or len(cmd_records) == 0:
        print(f"  Warning: Empty data in {bag_dir}")
        return None, None, None, None

    state_times = np.array([r[0] for r in state_records])
    states_list = [r[1] for r in state_records]
    cmd_times = np.array([r[0] for r in cmd_records])
    commands_list = [r[1] for r in cmd_records]

    return states_list, state_times, commands_list, cmd_times


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
        if result[0] is not None:
            extracted.append(result)

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


def build_output_path(args, include_states: bool, split: str | None = None) -> str:
    states_suffix = "_with_states" if include_states else ""
    split_suffix = f"_{split}" if split else ""
    filename = (
        f"data_odrive_new_estimator_history_{args.history}"
        f"_interval_{args.interval}_delay_{args.delay}"
        f"_horizon{args.horizon}{states_suffix}{split_suffix}.npz"
    )
    return str(Path(__file__).resolve().parent / "data" / filename)


def main():
    parser = argparse.ArgumentParser(description="Create actuator model dataset from .db3 trajectory bags")
    parser.add_argument(
        "--data_dir",
        default=str(Path(__file__).resolve().parents[2] / "logs" / "actuator_model" / "data" / "train_traj"),
        help="Path to directory containing trajectory subdirectories (optionally with train/ and validation/ subfolders)",
    )
    parser.add_argument("--output", default=None, help="Output .npz file path (auto-generated if omitted; ignored when train/validation split is detected)")
    parser.add_argument("--history", type=int, default=10, help="Number of past command/state steps (default: 10)")
    parser.add_argument("--interval", type=float, default=0.02, help="Time between history steps in seconds (default: 0.02)")
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
            output_path = build_output_path(args, include_states, split=split)
            process_split(split_dir, output_path, args, include_states)
    else:
        output_path = args.output if args.output else build_output_path(args, include_states)
        process_split(data_dir, output_path, args, include_states)


if __name__ == "__main__":
    main()

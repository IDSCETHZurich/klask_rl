# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot a saved debug_sim2real rollout.

Loads the .npz produced by debug_sim2real.py and plots every observation
channel plus the policy actions. No Isaac Sim required.

Usage:
    python debug_sim2real_plot.py [npz_path]

If `npz_path` is omitted, the most recently modified .npz under
``logs/rl_games/klask/debug_sim2real/`` is used.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


# Base observation layout (first 12 dims).
# Mirrors source/klask_rl/.../env_cfg/klask_rl_observations_cfg.py:51-82
BASE_OBS_GROUPS = [
    ("player pos [m]",   slice(0, 2),   ("x", "y")),
    ("player vel [m/s]", slice(2, 4),   ("x", "y")),
    ("opp pos [m]",      slice(4, 6),   ("x", "y")),
    ("opp vel [m/s]",    slice(6, 8),   ("x", "y")),
    ("ball pos [m]",     slice(8, 10),  ("x", "y")),
    ("ball vel [m/s]",   slice(10, 12), ("x", "y")),
]

# Extended observation layout (dims 12-19, only present when obs_dim >= 20).
# Mirrors klask_rl_observations_cfg.py extended group.
# (label, is_angle) — angles are stored as radians and plotted as degrees.
EXTENDED_OBS_SPEC = [
    ("angle own→ball→opp_goal [deg]", True),
    ("angle opp→ball→own_goal [deg]", True),
    ("angle own→ball→opp [deg]",      True),
    ("angle opp→ball→own [deg]",      True),
    ("dist ball→own_goal [m]",        False),
    ("dist ball→opp_goal [m]",        False),
    ("dist ball→own_peg [m]",         False),
    ("dist ball→opp_peg [m]",         False),
]


def find_latest_npz() -> str:
    candidates = glob.glob(
        os.path.join("logs", "rl_games", "klask", "debug_sim2real", "*.npz")
    )
    if not candidates:
        raise FileNotFoundError(
            "No .npz files found under logs/rl_games/klask/debug_sim2real/. "
            "Run debug_sim2real.py first, or pass a path explicitly."
        )
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz_path",
        nargs="?",
        default=None,
        help="Path to the .npz produced by debug_sim2real.py. "
             "Defaults to the most recent one in logs/rl_games/klask/debug_sim2real/.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for PNG files. Defaults to a folder next to the .npz "
             "named <npz_basename>_plots/.",
    )
    args = parser.parse_args()

    npz_path = args.npz_path or find_latest_npz()
    print(f"[INFO] loading: {npz_path}")
    data = np.load(npz_path)

    obs_full = data["obs_full"]    # (T, obs_dim)
    action = data["action"]        # (T, action_dim)
    player_init_pos = data["player_init_pos"]
    player_init_vel = data["player_init_vel"]
    opponent_fixed_pos = data["opponent_fixed_pos"]
    ball_fixed_pos = data["ball_fixed_pos"]

    T, obs_dim = obs_full.shape
    steps = np.arange(T)

    out_dir = args.out_dir or (os.path.splitext(npz_path)[0] + "_plots")
    os.makedirs(out_dir, exist_ok=True)

    title_suffix = (
        f"T={T}, obs_dim={obs_dim} | "
        f"player init pos=({player_init_pos[0]:.3f}, {player_init_pos[1]:.3f}) "
        f"vel=({player_init_vel[0]:.3f}, {player_init_vel[1]:.3f}) | "
        f"opp=({opponent_fixed_pos[0]:.3f}, {opponent_fixed_pos[1]:.3f}) | "
        f"ball=({ball_fixed_pos[0]:.3f}, {ball_fixed_pos[1]:.3f})"
    )

    def save_panel(name: str, ylabel: str, traces: list[tuple[str, np.ndarray]]):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for label, y in traces:
            ax.plot(steps, y, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("step")
        if len(traces) > 1 or traces[0][0]:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} | {title_suffix}", fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[INFO] saved: {out_path}")

    # --- Action ---
    action_labels = ["action_x", "action_y"] if action.shape[1] == 2 else [
        f"action[{i}]" for i in range(action.shape[1])
    ]
    save_panel(
        "00_action",
        "action [-1, 1]",
        [(label, action[:, i]) for i, label in enumerate(action_labels)],
    )

    # --- Base obs groups ---
    base_filenames = [
        "01_player_pos",
        "02_player_vel",
        "03_opp_pos",
        "04_opp_vel",
        "05_ball_pos",
        "06_ball_vel",
    ]
    for fname, (ylabel, sl, comp_labels) in zip(base_filenames, BASE_OBS_GROUPS):
        chunk = obs_full[:, sl]
        save_panel(
            fname,
            ylabel,
            [(comp, chunk[:, j]) for j, comp in enumerate(comp_labels)],
        )

    # --- Extended obs (one image per dim; angles converted rad → deg) ---
    extended_dims = obs_dim - 12 if obs_dim >= 20 else 0
    extended_dims = min(extended_dims, len(EXTENDED_OBS_SPEC))
    for k in range(extended_dims):
        label, is_angle = EXTENDED_OBS_SPEC[k]
        values = obs_full[:, 12 + k]
        if is_angle:
            values = np.rad2deg(values)
        # filename-safe slug
        slug = (
            label.split(" [")[0]
            .replace("→", "_to_")
            .replace(" ", "_")
        )
        save_panel(f"{7 + k:02d}_{slug}", label, [("", values)])

    print(f"[INFO] all plots written under: {out_dir}")


if __name__ == "__main__":
    main()

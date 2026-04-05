#!/usr/bin/env python3
"""Generate a GIF showing pegs moving across all board positions."""

import os
import sys

import cv2
import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRITE_DIR = os.path.join(PKG_DIR, "assets", "sprites")
BG_PATH = os.path.join(PKG_DIR, "assets", "background", "median_background.png")
OUT_PATH = os.path.join(PKG_DIR, "debug", "test_renders", "demo.gif")

if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from board_renderer import BoardRenderer


def raster_path(xs, ys):
    """Snake/raster scan: left-right on even rows, right-left on odd rows."""
    waypoints = []
    for i, y in enumerate(ys):
        row_xs = xs if i % 2 == 0 else xs[::-1]
        for x in row_xs:
            waypoints.append((x, y))
    return waypoints


def interpolate_path(waypoints, steps_per_segment):
    """Linearly interpolate between waypoints for smooth motion."""
    points = []
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    points.append(waypoints[-1])
    return points


def _render_gif(renderer, left_path, right_path, ball_path, out_path):
    """Render frames and write a GIF."""
    from PIL import Image

    n_frames = len(left_path)
    frames = []
    for i in range(n_frames):
        img = renderer.render(left_path[i], right_path[i], ball_path[i])
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=50,
        loop=0,
    )
    print(f"Saved {len(frames)}-frame GIF to {out_path}")


def main():
    """Generate demo GIFs of pegs moving across the board."""
    steps_per_segment = 6

    # ------------------------------------------------------------------
    # Camera-frame demo
    # ------------------------------------------------------------------
    cam_renderer = BoardRenderer(
        sprite_dir=SPRITE_DIR,
        background_path=BG_PATH,
        output_size=(210, 160),
        fast_mode=False,
        target_frame="camera",
    )

    left_xs = np.linspace(0.02, 0.17, 8)
    left_ys = np.linspace(0.02, 0.30, 6)
    left_path = interpolate_path(raster_path(left_xs, left_ys), steps_per_segment)

    right_xs = np.linspace(0.25, 0.40, 8)
    right_ys = np.linspace(0.02, 0.30, 6)
    right_path = interpolate_path(raster_path(right_xs, right_ys), steps_per_segment)

    n_frames = len(left_path)
    ball_cx, ball_cy = 0.21, 0.16
    ball_r = 0.06
    ball_angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    ball_path = [(ball_cx + ball_r * np.cos(a), ball_cy + ball_r * np.sin(a)) for a in ball_angles]

    _render_gif(cam_renderer, left_path, right_path, ball_path, OUT_PATH)

    # ------------------------------------------------------------------
    # Sim-frame demo (origin at centre, X right, Y up)
    # ------------------------------------------------------------------
    sim_renderer = BoardRenderer(
        sprite_dir=SPRITE_DIR,
        background_path=BG_PATH,
        output_size=(160, 210),
        fast_mode=False,
        target_frame="sim",
    )

    # Player peg (left_peg): raster over bottom half (sim_y < 0)
    # sim_x in [-0.14, 0.14], sim_y in [-0.19, -0.04]
    player_sxs = np.linspace(-0.14, 0.14, 6)
    player_sys = np.linspace(-0.19, -0.04, 8)
    player_path = interpolate_path(raster_path(player_sxs, player_sys), steps_per_segment)

    # Opponent peg (right_peg): raster over top half (sim_y > 0)
    # sim_x in [-0.14, 0.14], sim_y in [0.04, 0.19]
    opp_sxs = np.linspace(-0.14, 0.14, 6)
    opp_sys = np.linspace(0.04, 0.19, 8)
    opp_path = interpolate_path(raster_path(opp_sxs, opp_sys), steps_per_segment)

    n_frames_sim = len(player_path)
    ball_r_sim = 0.06
    ball_angles_sim = np.linspace(0, 2 * np.pi, n_frames_sim, endpoint=False)
    ball_path_sim = [(ball_r_sim * np.cos(a), ball_r_sim * np.sin(a)) for a in ball_angles_sim]

    sim_out = OUT_PATH.replace(".gif", "_sim.gif")
    _render_gif(sim_renderer, player_path, opp_path, ball_path_sim, sim_out)


if __name__ == "__main__":
    main()

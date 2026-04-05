#!/usr/bin/env python3
"""Test and benchmark the BoardRenderer."""

import os
import sys
import time

import numpy as np

# Paths
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRITE_DIR = os.path.join(PKG_DIR, "assets", "sprites")
BG_PATH = os.path.join(PKG_DIR, "assets", "background", "median_background.png")
OUT_DIR = os.path.join(PKG_DIR, "debug", "test_renders")

if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from board_renderer import BoardRenderer


def _benchmark(renderer, positions, n_iters=10000):
    """Run a rendering benchmark and print stats."""
    lp, rp, bp = positions
    for _ in range(100):
        renderer.render(lp, rp, bp)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        renderer.render(lp, rp, bp)
        times.append(time.perf_counter() - t0)
    times_ms = np.array(times) * 1000
    print(f"  Benchmark ({n_iters} iters):")
    print(f"    median: {np.median(times_ms):.4f} ms")
    print(f"    mean:   {np.mean(times_ms):.4f} ms")
    print(f"    p95:    {np.percentile(times_ms, 95):.4f} ms")
    print(f"    p99:    {np.percentile(times_ms, 99):.4f} ms")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Camera-frame test positions (metres, origin top-left, X right, Y down)
    cam_test_cases = [
        {
            "name": "centre",
            "left_peg": (0.10, 0.16),
            "right_peg": (0.32, 0.16),
            "ball": (0.21, 0.16),
        },
        {
            "name": "corners",
            "left_peg": (0.03, 0.03),
            "right_peg": (0.39, 0.29),
            "ball": (0.04, 0.28),
        },
        {
            "name": "near_goals",
            "left_peg": (0.10, 0.16),
            "right_peg": (0.32, 0.16),
            "ball": (0.10, 0.16),
        },
    ]

    # Sim-frame equivalents (origin at centre, X right, Y up)
    # sim_x = cam_y - 0.16, sim_y = cam_x - 0.21
    sim_test_cases = [
        {
            "name": "centre",
            "left_peg": (0.0, -0.11),
            "right_peg": (0.0, 0.11),
            "ball": (0.0, 0.0),
        },
        {
            "name": "corners",
            "left_peg": (-0.13, -0.18),
            "right_peg": (0.13, 0.18),
            "ball": (0.12, -0.17),
        },
        {
            "name": "near_goals",
            "left_peg": (0.0, -0.11),
            "right_peg": (0.0, 0.11),
            "ball": (0.0, -0.11),
        },
    ]

    import cv2

    # ------------------------------------------------------------------
    # Camera frame tests
    # ------------------------------------------------------------------
    for fast_mode in [False, True]:
        mode_name = "fast" if fast_mode else "precise"
        print(f"\n=== Camera | Mode: {mode_name} ===")

        for size_name, output_size in [("126x96", (126, 96)), ("63x48", (63, 48))]:
            print(f"\n  Output size: {size_name}")

            t0 = time.perf_counter()
            renderer = BoardRenderer(
                sprite_dir=SPRITE_DIR,
                background_path=BG_PATH,
                output_size=output_size,
                fast_mode=fast_mode,
                target_frame="camera",
            )
            init_ms = (time.perf_counter() - t0) * 1000
            print(f"  Init time: {init_ms:.1f} ms")

            for tc in cam_test_cases:
                img = renderer.render(tc["left_peg"], tc["right_peg"], tc["ball"])
                fname = f"cam_{mode_name}_{size_name}_{tc['name']}.png"
                cv2.imwrite(os.path.join(OUT_DIR, fname), img)
                print(f"  Saved: {fname}  shape={img.shape}  dtype={img.dtype}")

            _benchmark(renderer, ((0.10, 0.16), (0.32, 0.16), (0.21, 0.16)))

    # ------------------------------------------------------------------
    # Sim frame tests
    # ------------------------------------------------------------------
    for fast_mode in [False, True]:
        mode_name = "fast" if fast_mode else "precise"
        print(f"\n=== Sim | Mode: {mode_name} ===")

        for size_name, output_size in [("96x126", (96, 126)), ("48x63", (48, 63))]:
            print(f"\n  Output size: {size_name}")

            t0 = time.perf_counter()
            renderer = BoardRenderer(
                sprite_dir=SPRITE_DIR,
                background_path=BG_PATH,
                output_size=output_size,
                fast_mode=fast_mode,
                target_frame="sim",
            )
            init_ms = (time.perf_counter() - t0) * 1000
            print(f"  Init time: {init_ms:.1f} ms")

            for tc in sim_test_cases:
                img = renderer.render(tc["left_peg"], tc["right_peg"], tc["ball"])
                fname = f"sim_{mode_name}_{size_name}_{tc['name']}.png"
                cv2.imwrite(os.path.join(OUT_DIR, fname), img)
                print(f"  Saved: {fname}  shape={img.shape}  dtype={img.dtype}")

            _benchmark(renderer, ((0.0, -0.11), (0.0, 0.11), (0.0, 0.0)))


if __name__ == "__main__":
    main()

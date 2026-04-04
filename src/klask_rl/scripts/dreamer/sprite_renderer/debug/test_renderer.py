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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Test positions (metres)
    test_cases = [
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

    for fast_mode in [False, True]:
        mode_name = "fast" if fast_mode else "precise"
        print(f"\n=== Mode: {mode_name} ===")

        for size_name, output_size in [("126x96", (126, 96)), ("63x48", (63, 48))]:
            print(f"\n  Output size: {size_name}")

            t0 = time.perf_counter()
            renderer = BoardRenderer(
                sprite_dir=SPRITE_DIR,
                background_path=BG_PATH,
                output_size=output_size,
                fast_mode=fast_mode,
            )
            init_ms = (time.perf_counter() - t0) * 1000
            print(f"  Init time: {init_ms:.1f} ms")

            # Render test cases and save
            import cv2

            for tc in test_cases:
                img = renderer.render(tc["left_peg"], tc["right_peg"], tc["ball"])
                fname = f"{mode_name}_{size_name}_{tc['name']}.png"
                cv2.imwrite(os.path.join(OUT_DIR, fname), img)
                print(f"  Saved: {fname}  shape={img.shape}  dtype={img.dtype}")

            # Benchmark
            n_iters = 10000
            lp = (0.10, 0.16)
            rp = (0.32, 0.16)
            bp = (0.21, 0.16)

            # Warm up
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


if __name__ == "__main__":
    main()

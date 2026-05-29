#!/usr/bin/env python3
"""Visualize sprite-renderer augmentations as an (axis x level) grid.

Loads an augmentation YAML (same schema as the training config's
``augmentation`` block), builds ONE renderer with the full augmentation
config (exactly as training does), and renders selected variants by their
``aug_id``. Each row sweeps one enabled axis while holding the other axes
at their identity-closest level; the ``aug_id`` for each cell is printed
under the image so you can map back to the linear index used by the
renderer at runtime.

Run:
    /workspace/isaaclab/_isaac_sim/python.sh \
        /workspace/klask_rl/scripts/dreamer/sprite_renderer/debug/visualize_augmentations.py
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPRITE_DIR = os.path.join(PKG_DIR, "assets", "sprites")
BG_PATH = os.path.join(PKG_DIR, "assets", "background", "median_background.png")
OUT_DIR = os.path.join(PKG_DIR, "debug", "test_renders")

if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from board_renderer import BoardRenderer, _AUG_AXES, _axis_levels  # noqa: E402

# Fixed sim-frame test positions (metres, origin at board centre, X right, Y up).
LEFT_PEG = (0.0, -0.10)
RIGHT_PEG = (0.0, 0.10)
BALL = (0.05, 0.05)

# Value of each axis that produces the identity transform.
_IDENTITY_VALUE = {
    "brightness": 1.0,
    "contrast": 1.0,
    "gamma": 1.0,
    "color_temp": 0.0,
    "saturation": 1.0,
    "hue": 0.0,
}


def _closest_idx(levels, target: float) -> int:
    """Index of the level whose value is closest to ``target``."""
    return int((abs(levels - target)).argmin())


def _combo_to_aug_id(level_indices: tuple[int, ...], level_counts: tuple[int, ...]) -> int:
    """Map a per-axis level-index tuple to the linear aug_id in the renderer.

    The renderer builds variants in ``itertools.product(*level_lists)`` order
    over the enabled axes (in canonical ``_AUG_AXES`` order). product iterates
    the LAST axis fastest, so:
        aug_id = 1 + sum_k (idx_k * prod_{j > k} len_j)
    The ``+1`` accounts for variant 0 being identity.
    """
    linear = 0
    for k, idx in enumerate(level_indices):
        stride = 1
        for j in range(k + 1, len(level_counts)):
            stride *= level_counts[j]
        linear += idx * stride
    return 1 + linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "augmentation_example.yaml"),
        help="Path to augmentation YAML.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(OUT_DIR, "augmentations.png"),
        help="Output figure path.",
    )
    parser.add_argument("--out_w", type=int, default=192)
    parser.add_argument("--out_h", type=int, default=252)
    args = parser.parse_args()

    with open(args.config) as f:
        aug_cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Enabled axes in canonical order (matches the renderer's product order).
    enabled = [
        (axis, _axis_levels(aug_cfg[axis])) for axis in _AUG_AXES if axis in aug_cfg and aug_cfg[axis].get("enabled")
    ]
    if not enabled:
        raise SystemExit(f"No axes enabled in {args.config}; nothing to visualize.")

    level_counts = tuple(len(levels) for _, levels in enabled)
    identity_idx = {axis: _closest_idx(levels, _IDENTITY_VALUE[axis]) for axis, levels in enabled}

    # Single renderer with the FULL augmentation config — same as training.
    renderer = BoardRenderer(
        sprite_dir=SPRITE_DIR,
        background_path=BG_PATH,
        output_size=(args.out_w, args.out_h),
        fast_mode=False,
        target_frame="sim",
        augmentation_cfg=aug_cfg,
    )
    print(f"Renderer built with {renderer.num_variants} variants ({' x '.join(map(str, level_counts))} + 1).")

    n_rows = len(enabled)
    n_cols = max(level_counts)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.suptitle(
        f"Sprite-renderer augmentations — {os.path.basename(args.config)} — {renderer.num_variants} variants",
        fontsize=11,
    )

    for row, (axis, levels) in enumerate(enabled):
        # Sweep this axis; hold the rest at their identity-closest level.
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(levels):
                ax.axis("off")
                continue
            indices = tuple(col if name == axis else identity_idx[name] for name, _ in enabled)
            aug_id = _combo_to_aug_id(indices, level_counts)
            img = renderer.render(LEFT_PEG, RIGHT_PEG, BALL, aug_id=aug_id)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if col == 0:
                ax.set_ylabel(axis, fontsize=12, fontweight="bold")
            ax.set_title(f"id={aug_id}\n{axis}={levels[col]:.3g}", fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved: {args.out}  shape={n_rows}x{n_cols + 1}")


if __name__ == "__main__":
    main()

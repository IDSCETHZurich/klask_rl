#!/usr/bin/env python3
"""Export a Dreamer training checkpoint to inference-only format.

Strips optimizer state, value/reward/cont/decoder heads, and frozen copies,
keeping only encoder, RSSM, and actor weights. Also exports the model config
as a JSON file for use by DreamerInference.

Usage:
    python export_dreamer_checkpoint.py \
        --checkpoint /path/to/latest.pt \
        --config /path/to/config.yaml \
        --output-dir /path/to/output_dir

This produces, under output-dir:
    - <checkpoint_stem>_inference.pt  (~150-200 MB vs ~552 MB full checkpoint)
    - <checkpoint_stem>_inference_config.json   (model architecture config)
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Allowlisted config fields per inference-side module. Mirrors the YAML schema
# in r2dreamer/configs/model/_base_.yaml; audit against that file when the
# training config grows new keys. `device` is intentionally omitted — the
# inference runtime sets its own device at load time.
INFERENCE_FIELDS = {
    "rssm": frozenset({
        "stoch",
        "deter",
        "hidden",
        "discrete",
        "img_layers",
        "obs_layers",
        "dyn_layers",
        "blocks",
        "act",
        "norm",
        "unimix_ratio",
        "initial",
    }),
    "encoder": frozenset({"mlp_keys", "cnn_keys", "mlp", "cnn"}),
    "mlp": frozenset({
        "shape",
        "layers",
        "units",
        "act",
        "norm",
        "outscale",
        "symlog_inputs",
        "name",
    }),
    "cnn": frozenset({
        "act",
        "norm",
        "kernel_size",
        "minres",
        "depth",
        "mults",
    }),
    "actor": frozenset({
        "shape",
        "layers",
        "units",
        "act",
        "norm",
        "dist",
        "outscale",
        "symlog_inputs",
        "name",
    }),
    # Union across dist variants (bounded_normal / onehot / multi_onehot / symexp_twohot).
    "actor_dist": frozenset({
        "name",
        "min_std",
        "max_std",
        "unimix_ratio",
        "shape",
        "bin_num",
    }),
}


def _filter_to_fields(data: dict, allowed: frozenset) -> dict:
    """Keep only keys in the allowlist."""
    return {k: v for k, v in (data or {}).items() if k in allowed}


def _parse_opponent_separation(resolved: dict) -> bool:
    """Return whether the training run used opponent_separation.

    r2dreamer's training config exposes `opponent_separation` either as a scalar
    bool or as a nested struct with an `enabled` key. Mirror the unpacking done
    in evaluate_dreamer.py so the inference JSON carries a single bool.
    """
    raw = resolved.get("opponent_separation", False)
    if isinstance(raw, dict):
        return bool(raw.get("enabled", False))
    return bool(raw)


def build_inference_config(resolved: dict) -> dict:
    """Project a resolved training config onto the inference JSON schema."""
    model = resolved.get("model", resolved) or {}
    env = resolved.get("env", {}) or {}

    rssm = _filter_to_fields(model.get("rssm"), INFERENCE_FIELDS["rssm"])

    enc_src = model.get("encoder") or {}
    encoder = _filter_to_fields(enc_src, INFERENCE_FIELDS["encoder"])
    encoder.setdefault("mlp_keys", "$^")
    encoder.setdefault("cnn_keys", "image")
    encoder["mlp"] = _filter_to_fields(enc_src.get("mlp"), INFERENCE_FIELDS["mlp"])
    encoder["cnn"] = _filter_to_fields(enc_src.get("cnn"), INFERENCE_FIELDS["cnn"])

    actor_src = model.get("actor") or {}
    actor = _filter_to_fields(actor_src, INFERENCE_FIELDS["actor"])

    # Training configs expose cont/disc/multi_disc dist variants; inference is continuous.
    dist_src = actor_src.get("dist") or {}
    if isinstance(dist_src, dict) and "cont" in dist_src:
        dist_src = dist_src["cont"]
    actor["dist"] = _filter_to_fields(dist_src, INFERENCE_FIELDS["actor_dist"])

    # Training leaves actor.shape null and lets the env infer it; Klask is 2D continuous.
    if actor.get("shape") in (None, []):
        actor["shape"] = [2]

    env_size = env.get("size", [64, 64])
    image_size = int(env_size[0]) if isinstance(env_size, (list, tuple)) else int(env_size)
    max_velocity = float(env.get("max_velocity", 0.6))
    obs_mode = "image" if encoder["mlp_keys"] == "$^" else "image_and_state"

    return {
        "image_size": image_size,
        "obs_mode": obs_mode,
        "max_velocity": max_velocity,
        "opponent_separation": _parse_opponent_separation(resolved),
        "rssm": rssm,
        "encoder": encoder,
        "actor": actor,
    }


def main():
    """Main function to export Dreamer checkpoint."""
    parser = argparse.ArgumentParser(description="Export Dreamer checkpoint for inference.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Hydra training config YAML (optional, for extracting model params)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write <stem>_inference.pt and <stem>_config.json into",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_path.stem}_inference.pt"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "agent_state_dict" not in checkpoint:
        print("Error: Checkpoint does not contain 'agent_state_dict' key.", file=sys.stderr)
        print(f"Available keys: {list(checkpoint.keys())}", file=sys.stderr)
        sys.exit(1)

    full_state_dict = checkpoint["agent_state_dict"]
    full_size = sum(v.numel() * v.element_size() for v in full_state_dict.values())

    inference_prefixes = ("encoder.", "rssm.", "actor.")
    inference_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith(inference_prefixes)}
    inference_size = sum(v.numel() * v.element_size() for v in inference_state_dict.values())

    print(f"Full checkpoint: {len(full_state_dict)} keys, {full_size / 1e6:.1f} MB")
    print(f"Inference-only:  {len(inference_state_dict)} keys, {inference_size / 1e6:.1f} MB")
    print(f"Reduction:       {(1 - inference_size / full_size) * 100:.1f}%")

    torch.save(inference_state_dict, output_path)
    print(f"Saved inference checkpoint: {output_path}")

    if not args.config:
        print("No --config provided, skipping config JSON export.")
        print("DreamerInference will use default_size50m() configuration.")
        return

    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("Warning: omegaconf not installed, skipping config export.")
        print("You can manually create a dreamer_config.json or rely on default_size50m().")
        return

    # Hydra's saved config embeds `${now:...}` in logdir; stub the resolver so
    # full-tree resolution doesn't fail on fields the script never reads.
    OmegaConf.register_new_resolver("now", lambda *_: "", replace=True)

    cfg = OmegaConf.load(args.config)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    config_dict = build_inference_config(resolved)

    config_path = output_dir / f"{checkpoint_path.stem}_inference_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()

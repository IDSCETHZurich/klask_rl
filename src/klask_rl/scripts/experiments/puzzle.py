"""Collect held-out trajectories or play the ID exact/random/zero tournament."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parents[1]


def experiment_config(base, mode, checkpoint, config, provenance, timing):
    result = json.loads(json.dumps(base))
    result["agents"]["Dreamer-lineage"] = {
        "kind": "dreamer",
        "checkpoint": str(checkpoint),
        "config": str(config),
    }
    result["matches"] = []
    conditions = ("exact",) if mode == "collect" else ("exact", "random", "zero")
    opponents = ("Dreamer-lineage", "PPO-B") if mode == "collect" else ("Dreamer-lineage",)
    for opponent in opponents:
        for condition in conditions:
            domain = "id" if opponent == "Dreamer-lineage" else "ood"
            match = {
                "id": f"{domain}_{condition}",
                "a": "D-RSSM",
                "b": opponent,
                "condition": condition,
                "primary": condition == "zero",
                "action_timing": timing,
                "agent_conditions": {"D-RSSM": condition, "Dreamer-lineage": "zero"},
            }
            if mode == "collect":
                match.update(trajectory_domain=domain, trajectory_agent="D-RSSM")
            result["matches"].append(match)
    result["experiment"] = {
        "mode": mode,
        "id_provenance": provenance,
        "held_out": "new simulation seeds, no training or checkpoint updates",
        "fixed_opponent_input": "zero",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "gameplay"))
    parser.add_argument("--config", type=Path, default=SCRIPTS / "tournament/config.yaml")
    parser.add_argument("--id-checkpoint", type=Path, required=True)
    parser.add_argument("--id-config", type=Path, required=True)
    parser.add_argument(
        "--id-provenance",
        required=True,
        help="How this opponent relates to the training population; say proxy if unverified",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--games",
        type=int,
        default=64,
        help="Per domain for collection; per condition for gameplay",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--timing", choices=("aligned", "legacy"), default="aligned")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = experiment_config(
        yaml.safe_load(args.config.read_text()),
        args.mode,
        args.id_checkpoint.resolve(),
        args.id_config.resolve(),
        args.id_provenance,
        args.timing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_name(args.output.name + "_config.yaml")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [
        sys.executable,
        str(SCRIPTS / "tournament/run_tournament.py"),
        "--config",
        str(config_path),
        "--output",
        str(args.output),
        "--games",
        str(args.games),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--devices",
        *args.devices,
    ]
    if args.preflight_only:
        command.append("--preflight-only")
    subprocess.run(command, check=True)
    if not args.preflight_only:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "experiments/analyze.py"),
                "--input",
                str(args.output),
                "--output",
                str(args.output / "analysis"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

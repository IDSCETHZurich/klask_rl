"""Train three scratch seeds on the last two visible GPUs; evaluate saved curves separately."""

import argparse
import csv
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "tournament"))
from raw_data import utc_now, write_csv, write_json
from run_tournament import resolve_path, sha256
from scheduler import run_jobs


def devices_for_run(explicit=None):
    import torch

    count = torch.cuda.device_count()
    devices = explicit if explicit else [f"cuda:{i}" for i in range(max(0, count - 2), count)]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("Expose two GPUs or specify exactly two distinct --devices cuda:N cuda:M")
    for device in devices:
        if not device.startswith("cuda:") or not device[5:].isdigit() or int(device[5:]) >= count:
            raise ValueError(f"Unavailable device {device}; container sees {count} GPUs")
    return devices


def scratch_ppo_config(source, root, seed, device, num_envs, max_steps):
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    params = cfg["params"]
    if params["network"]["mlp"]["units"] != [256, 128, 64]:
        raise ValueError("Expected reconstructed PPO-B 256/128/64 architecture")
    params.update(seed=seed, load_checkpoint=False, load_path="")
    train = params["config"]
    train.update(
        device=device,
        device_name=device,
        multi_gpu=False,
        num_actors=num_envs,
        max_epochs=-1,
        max_frames=max_steps or -1,
        save_frequency=0,
        save_best_after=10**12,
    )
    train.pop("score_to_win", None)
    if "self_play_config" in train:
        train["self_play_config"]["env_update_num"] = min(num_envs, train["self_play_config"]["env_update_num"])
    batch = num_envs * train["horizon_length"]
    if batch % train["minibatch_size"]:
        raise ValueError("num_envs * PPO horizon must be divisible by minibatch_size")
    actuator = cfg.get("env", {}).get("actuator_model", {})
    if actuator.get("enable"):
        actuator["checkpoint"] = str(resolve_path(actuator["checkpoint"], root))
    return cfg


def acquire_gpus(stack, devices):
    """Prevent these launchers sharing a GPU; unrelated external jobs remain visible in nvidia-smi."""
    import torch

    for device in sorted(devices):
        properties = torch.cuda.get_device_properties(device)
        identity = str(getattr(properties, "uuid", device)).replace(":", "_").replace("/", "_")
        stream = stack.enter_context(open(f"/tmp/klask-experiments-{identity}.lock", "w"))  # noqa: SIM115
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another experiment launcher already owns {device}; run PPO and FastSAC sequentially"
            ) from exc


def train(args):
    import torch

    root = args.project_root.resolve()
    output = args.output.resolve()
    devices = devices_for_run(args.devices)
    if args.hours <= 0 or args.num_envs <= 0 or args.max_env_steps < 0:
        raise ValueError("Positive hours/num-envs and nonnegative max-env-steps required")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("The baseline protocol requires three distinct seeds")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory; training resume is not a new independent seed")
    jobs = []
    for index, seed in enumerate(args.seeds):
        device = devices[index % len(devices)]
        directory = output / f"seed_{seed}"
        directory.mkdir()
        if args.algorithm == "ppo":
            cfg = scratch_ppo_config(args.ppo_config, root, seed, device, args.num_envs, args.max_env_steps)
            cfg_path = directory / "input_config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            command = [
                sys.executable,
                str(root / "scripts/rl_games/train_klask.py"),
                "--config",
                str(cfg_path),
                "--num_envs",
                str(args.num_envs),
                "--seed",
                str(seed),
                "--experiment-output",
                str(directory),
                "--wall-hours",
                str(args.hours),
                "--device",
                device,
                "--headless",
            ]
            quantum = args.num_envs * cfg["params"]["config"]["horizon_length"]
        else:
            quantum = args.num_envs
            iterations = (args.max_env_steps + quantum - 1) // quantum if args.max_env_steps else 2**31 - 1
            command = [
                sys.executable,
                str(root / "scripts/fast_sac/train_fast_sac_isaaclab.py"),
                "--num-envs",
                str(args.num_envs),
                "--seed",
                str(seed),
                "--experiment-output",
                str(directory),
                "--output-dir",
                str(directory / "trainer"),
                "--wall-hours",
                str(args.hours),
                "--save-interval",
                "0",
                "--num-learning-iterations",
                str(iterations),
                "--device",
                device,
                "--headless",
            ]
        jobs.append(
            {
                "id": f"{args.algorithm}_seed{seed}",
                "seed": seed,
                "device": device,
                "directory": str(directory),
                "command": command,
                "checkpoint_quantum": quantum,
                "first_checkpoint_at_9M": ((9_000_000 + quantum - 1) // quantum) * quantum,
            }
        )
    metadata = {
        "schema_version": 1,
        "experiment": "baseline_learning_curve",
        "algorithm": args.algorithm,
        "started_at": utc_now(),
        "argv": sys.argv,
        "seeds": args.seeds,
        "devices": devices,
        "jobs": jobs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_properties": {device: str(torch.cuda.get_device_properties(device)) for device in devices},
        "wall_hours_per_seed": args.hours,
        "initialization": "scratch",
        "complete": False,
        "source_sha256": {
            str(p.relative_to(root)): sha256(p)
            for folder in ("experiments", "rl_games", "fast_sac")
            for p in (root / "scripts" / folder).glob("*.py")
        },
    }
    write_json(output / "metadata.json", metadata)
    (output / "report.md").write_text(
        "# Baseline training\n\nThree independent scratch seeds. See seed_*/training.csv and checkpoints.csv.\n\n"
        "Training reward is not tournament score. Fixed-opponent scoring is a separate `baselines.py evaluate` run.\n\n"
        "9M denotes a requested threshold: use actual environment_steps in every comparison.\n\n"
        f"GPUs: {devices}. Budget: {args.hours} hours per seed, two seeds concurrently.\n"
    )
    for job in jobs:
        print(shlex.join(job["command"]), flush=True)
        print(
            f"First completed update reaching 9M: {job['first_checkpoint_at_9M']:,} transitions",
            flush=True,
        )
    if args.dry_run:
        return
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    def start(job):
        with (Path(job["directory"]) / "run.log").open("w") as log:
            return subprocess.Popen(job["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT)

    with ExitStack() as stack:
        acquire_gpus(stack, devices)
        try:
            run_jobs(
                jobs,
                start,
                lambda job, code: print(f"{job['id']}: exit {code}", flush=True),
            )
            metadata["complete"] = True
        finally:
            metadata["finished_at"] = utc_now()
            write_json(output / "metadata.json", metadata)


def evaluate(args):
    """Score every completed checkpoint against one explicitly selected fixed opponent."""
    devices = devices_for_run(args.devices)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((SCRIPTS / "tournament/config.yaml").read_text())
    if args.opponent_kind in ("dreamer", "ppo") and args.opponent_config is None:
        raise ValueError("Dreamer/PPO opponents require --opponent-config")
    opponent = {
        "kind": args.opponent_kind,
        "checkpoint": str(args.opponent_checkpoint.resolve()),
    }
    if args.opponent_config is not None:
        opponent["config"] = str(args.opponent_config.resolve())
    base["agents"]["Fixed-opponent"] = opponent
    evaluations = []
    for ledger in sorted(args.training.glob("seed_*/checkpoints.csv")):
        seed_dir = ledger.parent
        with ledger.open() as stream:
            checkpoints = list(csv.DictReader(stream))
        for row in checkpoints:
            if args.requested_steps and args.requested_steps not in map(
                int, filter(None, row["requested_steps"].split(";"))
            ):
                continue
            if sha256(row["checkpoint"]) != row["sha256"]:
                raise ValueError(f"Checkpoint changed: {row['checkpoint']}")
            name = "PPO-B" if row["algorithm"] == "ppo" else "FastSAC"
            cfg = json.loads(json.dumps(base))
            cfg["agents"][name]["checkpoint"] = row["checkpoint"]
            if name == "PPO-B":
                cfg["agents"][name]["config"] = str((seed_dir / "input_config.yaml").resolve())
            cfg["matches"] = [
                {
                    "id": "baseline",
                    "a": name,
                    "b": "Fixed-opponent",
                    "condition": "zero",
                    "primary": True,
                }
            ]
            cfg["experiment"] = {
                "training_seed": int(row["seed"]),
                "environment_steps": int(row["environment_steps"]),
                "requested_steps": row["requested_steps"],
                "initialization": "scratch",
                "algorithm": row["algorithm"],
                "checkpoint": row["checkpoint"],
            }
            directory = output / f"seed_{row['seed']}_step_{row['environment_steps']}"
            config_path = output / (directory.name + ".yaml")
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            command = [
                sys.executable,
                str(SCRIPTS / "tournament/run_tournament.py"),
                "--config",
                str(config_path),
                "--project-root",
                str(args.project_root.resolve()),
                "--output",
                str(directory),
                "--games",
                str(args.games),
                "--num-envs",
                str(args.num_envs),
                "--seed",
                str(args.eval_seed),
                "--devices",
                *devices,
                "--resume",
            ]
            with ExitStack() as stack:
                acquire_gpus(stack, devices)
                subprocess.run(command, check=True)
            evaluations.append(
                dict(
                    row,
                    evaluation_directory=directory.name,
                    opponent=str(args.opponent_checkpoint),
                    evaluation_seed=args.eval_seed,
                )
            )
            write_csv(output / "evaluations.csv", evaluations, list(evaluations[0]))
    if not evaluations:
        raise ValueError("No checkpoint matches selection; inspect seed_*/checkpoints.csv")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "experiments/analyze.py"),
            "--input",
            str(output),
            "--output",
            str(output / "analysis"),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        child = sub.add_parser(name)
        child.add_argument("--project-root", type=Path, default=PROJECT)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument(
            "--devices",
            nargs=2,
            help="Default: last two CUDA devices visible inside this container",
        )
        child.add_argument("--num-envs", type=int, default=512 if name == "train" else 64)
    training_parser = sub.choices["train"]
    training_parser.add_argument("--algorithm", choices=("ppo", "fast_sac"), required=True)
    training_parser.add_argument("--seeds", nargs=3, type=int, default=[0, 1, 2])
    training_parser.add_argument("--hours", type=float, default=36)
    training_parser.add_argument(
        "--max-env-steps",
        type=int,
        default=0,
        help="0: wall budget only; otherwise stop at first full update >= limit",
    )
    training_parser.add_argument(
        "--ppo-config",
        type=Path,
        default=SCRIPTS / "rl_games/config/klask_ppo_config_with_pretraining.yaml",
    )
    training_parser.add_argument("--dry-run", action="store_true")
    evaluation_parser = sub.choices["evaluate"]
    evaluation_parser.add_argument("--training", type=Path, required=True)
    evaluation_parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    evaluation_parser.add_argument("--opponent-config", type=Path)
    evaluation_parser.add_argument("--opponent-kind", choices=("dreamer", "ppo", "fast_sac"), default="dreamer")
    evaluation_parser.add_argument("--games", type=int, default=200)
    evaluation_parser.add_argument("--eval-seed", type=int, default=10000)
    evaluation_parser.add_argument(
        "--requested-steps",
        type=int,
        help="E.g. 9000000 for the matched-step checkpoint only",
    )
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()

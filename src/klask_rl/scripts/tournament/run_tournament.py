"""Plan, validate, run and resume the three-agent tournament in the Isaac Lab Python environment."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import pickle
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

MATCHES = [
    {
        "id": "drssm_ppo_exact",
        "a": "D-RSSM",
        "b": "PPO-B",
        "condition": "exact",
        "primary": False,
    },
    {
        "id": "drssm_ppo_zero",
        "a": "D-RSSM",
        "b": "PPO-B",
        "condition": "zero",
        "primary": True,
    },
    {
        "id": "drssm_fastsac_zero",
        "a": "D-RSSM",
        "b": "FastSAC",
        "condition": "zero",
        "primary": True,
    },
    {
        "id": "fastsac_ppo",
        "a": "FastSAC",
        "b": "PPO-B",
        "condition": "none",
        "primary": True,
    },
]


def selected_devices(args):
    devices = getattr(args, "devices", None)
    single = getattr(args, "device", None)
    if devices is not None and single is not None:
        raise ValueError("Use either --devices or --device")
    devices = list(devices) if devices is not None else ([single] if single else ["cuda:0", "cuda:1"])
    if not devices or any(re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None for device in devices):
        raise ValueError("Devices must be explicit CUDA indices, e.g. --devices cuda:0 cuda:1")
    if len(set(devices)) != len(devices):
        raise ValueError("Duplicate GPU indices would schedule two workers on the same GPU")
    return devices


def gpu_blockers(devices):
    import torch

    count = torch.cuda.device_count()
    missing = [device for device in devices if int(device.split(":")[1]) >= count]
    if missing:
        return [
            (
                f"Requested GPU(s) unavailable: {', '.join(missing)}; this container sees {count} CUDA GPU(s). "
                "Expose both GPUs to Docker, or use --devices cuda:0 for a single-GPU run."
            )
        ]
    return []


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, root):
    path = Path(value).expanduser()
    # Saved training configs use the container mount. Remap just that known
    # prefix; never search for a similarly named checkpoint or substitute one.
    if path.is_relative_to("/workspace/klask_rl"):
        path = root / path.relative_to("/workspace/klask_rl")
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def checkpoint_info(kind, path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if kind == "dreamer":
        state = ckpt["agent_state_dict"]
        action_dim = state["rssm._deter_net._dyn_in2.0.weight"].shape[1]
        if action_dim != 4:
            raise ValueError(f"D-RSSM requires a 4D RSSM input, found {action_dim}")
        return {
            "format": "agent_state_dict",
            "rssm_action_dim": action_dim,
            "training_step": int(ckpt["step"]),
        }
    if kind == "ppo":
        state = ckpt["model"]
        if tuple(state["a2c_network.actor_mlp.0.weight"].shape) != (256, 20):
            raise ValueError("Expected PPO-B's 20-input, 256/128/64 MLP")
        if tuple(state["a2c_network.mu.weight"].shape) != (2, 64):
            raise ValueError("Expected a two-action PPO-B policy")
        if state["running_mean_std.running_mean"].shape != (20,):
            raise ValueError("Missing PPO-B input normalization")
        return {
            "format": "rl_games model (normalization embedded)",
            "obs_dim": 20,
            "act_dim": 2,
        }
    if kind == "fast_sac":
        state = ckpt["actor_state_dict"]
        if state["net.0.weight"].shape[1] != 18 or state["fc_mu.weight"].shape[0] != 2:
            raise ValueError("Expected FastSAC 18-input/two-action actor")
        count = int(ckpt["obs_normalizer_state"]["count"].item())
        if count <= 0 or not ckpt["config"].get("obs_normalization", False):
            raise ValueError("FastSAC requires initialized observation normalization")
        return {
            "format": "actor_state_dict + obs_normalizer_state + config",
            "obs_dim": 18,
            "act_dim": 2,
            "normalizer_count": count,
            "training_config": ckpt["config"],
        }
    raise ValueError(f"Unsupported agent kind {kind}")


def build_plan(args):
    devices = selected_devices(args)
    selected = getattr(args, "matches", None)
    if selected and set(selected) - {m["id"] for m in MATCHES}:
        raise ValueError("Unknown matchup selection")
    matches = [dict(m) for m in MATCHES if not selected or m["id"] in selected]
    required_agents = {name for m in matches for name in (m["a"], m["b"])} | {"D-RSSM"}
    root = args.project_root.resolve()
    config = yaml.safe_load(args.config.read_text())
    games = args.games if args.games is not None else int(config["games"])
    num_envs = args.num_envs if args.num_envs is not None else int(config["num_envs"])
    seed = args.seed if args.seed is not None else int(config["seed"])
    duration = args.episode_length_s if args.episode_length_s is not None else float(config["episode_length_s"])
    if games < 2 or num_envs < 1 or duration <= 0 or int(config["bootstrap_samples"]) < 1:
        raise ValueError("Need at least two games, positive num_envs, episode length and bootstrap samples")
    expected = {"D-RSSM": "dreamer", "PPO-B": "ppo", "FastSAC": "fast_sac"}
    if {name: spec["kind"] for name, spec in config["agents"].items()} != expected:
        raise ValueError("Configure exactly D-RSSM, PPO-B and FastSAC; Stock DreamerV3 has no checkpoint yet")
    blockers, agents = [], {}
    for name, spec in config["agents"].items():
        if name not in required_agents:
            continue
        agent = dict(spec)
        for key in ("checkpoint", "config"):
            if key not in agent:
                continue
            path = resolve_path(agent[key], root)
            agent[key] = str(path)
            if not path.is_file():
                blockers.append(f"Missing {name} {key}: {path}")
            else:
                agent[key + "_sha256"] = sha256(path)
        if Path(agent["checkpoint"]).is_file():
            try:
                agent["inspection"] = checkpoint_info(agent["kind"], agent["checkpoint"])
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                ImportError,
                pickle.UnpicklingError,
            ) as exc:
                blockers.append(f"Invalid {name} checkpoint: {exc}")
        agents[name] = agent

    dreamer_path = Path(agents["D-RSSM"]["config"])
    evaluation_config = None
    if dreamer_path.is_file():
        cfg = OmegaConf.load(dreamer_path)
        if "defaults" in cfg:
            raise ValueError("Use the saved composed .hydra/config.yaml, not an uncomposed training config")
        separation = cfg.get("opponent_separation", False)
        enabled = separation.get("enabled", False) if OmegaConf.is_dict(separation) else separation
        if enabled is not True:
            raise ValueError("D-RSSM training config must enable opponent_separation")
        # Keep only evaluation dependencies, while resolving references against
        # the original config. No init_checkpoint or training load_path is used.
        cfg.device = devices[0]
        cfg.seed = seed
        evaluation_config = {
            key: OmegaConf.to_container(cfg[key], resolve=True) if OmegaConf.is_config(cfg[key]) else cfg[key]
            for key in ("model", "env", "opponent_separation")
        }
        evaluation_config.update(device=devices[0], seed=seed)
        evaluation_config["model"]["compile"] = False
        env = evaluation_config["env"]
        env["episode_length_s"] = duration
        env["device"] = devices[0]
        # Shared protocol: symmetric reset area, all scoring terminations,
        # deterministic dynamics parameters (no observation noise or DR).
        env["ball_reset_position_x"] = [-0.15, 0.15]
        env["ball_reset_position_y"] = [-0.1, 0.1]
        env["terminations"] = {
            name: True
            for name in (
                "time_out",
                "goal_scored",
                "goal_conceded",
                "player_in_goal",
                "opponent_in_goal",
            )
        }
        env["domain_randomization"] = {
            key: {"enable": False}
            for key in (
                "ball_mass",
                "material_ball",
                "material_board",
                "material_peg",
                "actuator",
            )
        }
        actuator = env.get("actuator_model", {})
        if actuator.get("enable"):
            path = resolve_path(args.actuator_checkpoint or actuator["checkpoint"], root)
            actuator["checkpoint"] = str(path)
            if not path.is_file():
                blockers.append(f"Missing actuator checkpoint (no fallback allowed): {path}")
            else:
                actuator["sha256"] = sha256(path)
        if env["task"] != "isaaclab_Klask-Rl-Dreamer-Sprite-v0":
            raise ValueError("This protocol is validated for the supplied sprite Dreamer training config")

    ppo_config = Path(agents["PPO-B"]["config"]) if "PPO-B" in agents else None
    if ppo_config is not None and ppo_config.is_file():
        ppo = yaml.safe_load(ppo_config.read_text())
        if ppo["params"]["network"]["mlp"]["units"] != [256, 128, 64]:
            raise ValueError("PPO-B config does not match the checkpoint's network")

    dependency_sources = {}
    for package, candidates, required in (
        (
            "r2dreamer",
            [
                root / "scripts/dreamer/r2dreamer",
                root.parent.parent / "third_party/r2dreamer",
            ],
            "dreamer.py",
        ),
        (
            "FastSAC",
            [
                root / "scripts/fast_sac/klask_her",
                root.parent.parent / "third_party/fast_sac",
            ],
            "klask_her/agents/fast_sac.py",
        ),
    ):
        if package == "FastSAC" and "FastSAC" not in required_agents:
            continue
        if not any((p / required).is_file() for p in candidates):
            blockers.append(
                f"Missing {package} submodule/mount; expected {required} under " + ", ".join(map(str, candidates))
            )
        else:
            package_root = next(p for p in candidates if (p / required).is_file())
            dependency_sources.update({str(p): sha256(p) for p in sorted(package_root.rglob("*.py"))})
    if importlib.util.find_spec("isaaclab") is None:
        blockers.append("Isaac Lab is unavailable in this Python interpreter; run inside the GPU container")
    assets = root / "scripts/dreamer/sprite_renderer/assets"
    if not (assets / "background/median_background.png").is_file():
        blockers.append(f"Missing sprite background under {assets}")
    for pattern in (
        "sprites/ball/ball_*.png",
        "sprites/peg/left_peg_*.png",
        "sprites/peg/right_peg_*.png",
    ):
        if not list(assets.glob(pattern)):
            blockers.append(f"Missing sprite assets: {assets / pattern}")

    jobs = []
    for match in matches:
        for leg in (0, 1):
            jobs.append(
                {
                    "id": f"{match['id']}_leg{leg}",
                    "match": match["id"],
                    "leg": leg,
                    "seat0": match["a"] if leg == 0 else match["b"],
                    "seat1": match["b"] if leg == 0 else match["a"],
                    "condition": match["condition"],
                    "seed": seed + leg,
                    "games": games // 2 + (games % 2 if leg == 0 else 0),
                    "device": devices[len(jobs) % len(devices)],
                }
            )
    # Source hashes work inside the container even when .git is not mounted.
    source_files = list((root / "scripts/tournament").glob("*.py"))
    source_files += list((root / "scripts/dreamer").glob("*.py"))
    source_files += list((root / "scripts/dreamer/sprite_renderer").rglob("*.py"))
    source_files += list((root / "source/klask_rl/klask_rl").rglob("*.py"))
    sources = {str(p.relative_to(root)): sha256(p) for p in sorted(source_files)}
    asset_hashes = {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(assets.rglob("*"))
        if p.is_file() and p.suffix in (".png", ".json", ".yaml")
    }
    versions = {}
    for package in ("torch", "numpy", "rl-games", "tensordict", "gymnasium", "isaaclab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    plan = {
        "schema_version": 1,
        "project_root": str(root),
        "agents": agents,
        "games_per_match": games,
        "num_envs": num_envs,
        "device": devices[0],
        "devices": devices,
        "evaluation_config": evaluation_config,
        "bootstrap_samples": int(config["bootstrap_samples"]),
        "bootstrap_seed": int(config["bootstrap_seed"]),
        "matches": matches,
        "jobs": jobs,
        "source_sha256": sources,
        "asset_sha256": asset_hashes,
        "dependency_source_sha256": dependency_sources,
        "package_versions": versions,
        "protocol": "balanced seats; fixed per-env episode quotas; aligned previous actions; post-reset observations",
    }
    plan["run_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan, blockers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    gpu_args = parser.add_mutually_exclusive_group()
    gpu_args.add_argument("--devices", nargs="+", help="Worker GPUs (default: cuda:0 cuda:1); one leg per GPU")
    gpu_args.add_argument("--device", help="Single-GPU shorthand, e.g. --device cuda:0")
    parser.add_argument("--games", type=int)
    parser.add_argument(
        "--matches",
        nargs="+",
        choices=[m["id"] for m in MATCHES],
        help="Run only these matchup/conditions; --games applies to each selected condition.",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episode-length-s", type=float)
    parser.add_argument("--actuator-checkpoint", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete legs; rerun partial legs from their seed",
    )
    args = parser.parse_args()
    plan, blockers = build_plan(args)
    blockers.extend(gpu_blockers(plan["devices"]))
    print(
        json.dumps(
            {"run_id": plan["run_id"], "jobs": plan["jobs"], "blockers": blockers},
            indent=2,
        )
    )
    if args.preflight_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "preflight.json").write_text(json.dumps({"plan": plan, "blockers": blockers}, indent=2) + "\n")
        return 1 if blockers else 0
    if blockers:
        raise SystemExit("Preflight failed; resolve the listed dependencies before simulation.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())["run_id"] != plan["run_id"]:
            raise SystemExit("Output already contains a run; use a new directory or --resume with matching settings.")
    elif any(output.glob("*/games.npz")):
        raise SystemExit("Output contains games without a manifest; use a new directory.")
    else:
        manifest_path.write_text(json.dumps(plan, indent=2) + "\n")
    from report import generate_report
    from scheduler import WorkerFailed, run_jobs

    signal.signal(signal.SIGTERM, signal.default_int_handler)
    pending = []
    for job in plan["jobs"]:
        job_dir = output / job["id"]
        job_dir.mkdir(exist_ok=True)
        result_path = job_dir / "games.npz"
        if args.resume and result_path.exists():
            import numpy as np

            with np.load(result_path, allow_pickle=False) as data:
                if (
                    str(data["meta_run_id"]) == plan["run_id"]
                    and str(data["meta_job_id"]) == job["id"]
                    and str(data["meta_complete"]) == "True"
                    and int(data["num_games"]) == job["games"]
                ):
                    continue
        pending.append(job)

    def start(job):
        job_dir = output / job["id"]
        command = [
            sys.executable,
            str(SCRIPT_DIR / "worker.py"),
            "--manifest",
            str(manifest_path),
            "--job",
            job["id"],
            "--headless",
            "--enable_cameras",
            "--device",
            job["device"],
        ]
        print(shlex.join(command), flush=True)
        print(f"Progress log: {job_dir / 'run.log'}", flush=True)
        with (job_dir / "run.log").open("w") as log:
            return subprocess.Popen(
                command, cwd=plan["project_root"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )

    def finished(job, returncode):
        print(f"{job['id']} on {job['device']}: exit {returncode}", flush=True)
        generate_report(output)

    try:
        run_jobs(pending, start, finished)
    except KeyboardInterrupt:
        return 130
    except WorkerFailed as exc:
        raise SystemExit(f"{exc}. Inspect the leg's run.log. Partial report saved.") from exc
    finally:
        # run_jobs has already stopped/reaped active workers, including on a
        # launch error. Collect the final partial files after that cleanup.
        report = generate_report(output)
    print((output / "report.md").read_text())
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

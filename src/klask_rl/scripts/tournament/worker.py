"""Execute one tournament leg. Launched by run_tournament.py inside Isaac Lab."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def configuration_for_job(plan, job):
    """Retarget every resolved model/environment device, not only env.device."""
    from omegaconf import OmegaConf

    device = job.get("device", plan["device"])

    def remap(value):
        if isinstance(value, dict):
            return {
                key: (
                    device
                    if key in ("device", "sim_device", "train_device")
                    else [device]
                    if key == "train_devices"
                    else remap(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [remap(item) for item in value]
        return value

    cfg = OmegaConf.create(remap(plan["evaluation_config"]))
    cfg.device = device
    cfg.env.device = device
    cfg.seed = job["seed"]
    cfg.env.seed = job["seed"]
    return cfg


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    # Same renderer limits as train_dreamer.py: each process must not enable
    # multi-GPU rendering while the other GPU is running a separate leg.
    sys.argv += ["--/renderer/multiGpu/enabled=false", "--/renderer/multiGpu/maxGpuCount=1"]
    launcher = AppLauncher(args)
    try:
        return run(args, launcher.app)
    finally:
        launcher.app.close()


def run(args, simulation_app):
    # All simulator-dependent imports must follow AppLauncher.
    import random
    import signal

    import gymnasium as gym
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from policies import DreamerPolicy, FeedForwardPolicy, episode_quotas
    from run_tournament import sha256
    from tqdm import tqdm

    if args.device.startswith("cuda:"):
        torch.cuda.set_device(args.device)
    plan = json.loads(args.manifest.read_text())
    job = next(j for j in plan["jobs"] if j["id"] == args.job)
    root = Path(plan["project_root"])
    sys.path.insert(0, str(root / "scripts/dreamer"))
    from evaluation import (
        _load_dreamer_agent,
        _load_fast_sac_opponent,
        _load_ppo_opponent,
        _make_eval_env,
    )
    from klask_rl.tasks.manager_based.klask_rl.eval_metrics import EvalMetricsTracker

    if args.device != job.get("device", plan["device"]):
        raise ValueError("Worker device differs from manifest")
    for path, expected in plan["source_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError(f"Source changed after preflight: {path}")
    for path, expected in plan["asset_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError(f"Sprite asset changed after preflight: {path}")
    for path, expected in plan["dependency_source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Dependency changed after preflight: {path}")
    # Check again immediately before loading: latest.pt may change during a run.
    for name in (job["seat0"], job["seat1"]):
        spec = plan["agents"][name]
        for key in ("checkpoint", "config"):
            if key in spec and sha256(spec[key]) != spec[key + "_sha256"]:
                raise ValueError(f"{name} {key} changed after preflight")
    cfg = configuration_for_job(plan, job)
    actuator = cfg.env.get("actuator_model", {})
    if actuator.get("enable") and sha256(actuator.checkpoint) != actuator.sha256:
        raise ValueError("Actuator checkpoint changed after preflight")
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.cuda.manual_seed_all(job["seed"])
    torch.set_float32_matmul_precision("high")
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    num_envs = min(plan["num_envs"], job["games"])
    env, _ = _make_eval_env(cfg.env, num_envs, tournament=True)
    tracker = None
    env_ids, episode_ids = [], []
    job_dir = args.manifest.parent / job["id"]
    job_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, job_dir / "evaluation_config.yaml")
    device = env.unwrapped.device
    quotas = episode_quotas(job["games"], num_envs, device)
    completed = torch.zeros(num_envs, dtype=torch.long, device=device)
    interrupted = False

    def save():
        if tracker is None:
            return
        temp = job_dir / "games.tmp.npz"
        tracker.save_npz(
            str(temp),
            metadata={
                "run_id": plan["run_id"],
                "job_id": job["id"],
                "seat0": job["seat0"],
                "seat1": job["seat1"],
                "condition": job["condition"],
                "seed": job["seed"],
                "requested_games": job["games"],
                "complete": tracker.total_games == job["games"],
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "cuda_version": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
            },
            extra_arrays={"env_id": env_ids, "episode_id": episode_ids},
        )
        temp.replace(job_dir / "games.npz")

    try:
        with torch.inference_mode():
            observations, _ = env.reset(seed=job["seed"])
            agents = []
            for seat, name in enumerate((job["seat0"], job["seat1"])):
                spec = plan["agents"][name]
                kind = spec["kind"]
                if kind == "dreamer":
                    spaces = dict(env.unwrapped.single_observation_space.spaces)
                    for flag in ("is_first", "is_terminal", "is_last"):
                        spaces[flag] = gym.spaces.Box(0, 1, (1,), dtype=bool)
                    model = _load_dreamer_agent(
                        cfg,
                        gym.spaces.Dict(spaces),
                        gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
                        spec["checkpoint"],
                        device,
                    )
                    agents.append(DreamerPolicy(model, num_envs, seat, job["condition"]))
                else:
                    if kind == "ppo":
                        obs_space = env.unwrapped.single_observation_space["policy" if seat == 0 else "opponent"]
                        model = _load_ppo_opponent(
                            spec["config"],
                            spec["checkpoint"],
                            num_envs,
                            device,
                            obs_space,
                        )
                    else:
                        model = _load_fast_sac_opponent(spec["checkpoint"], device)
                    agents.append(FeedForwardPolicy(model, kind, seat, observations))
            # rl_games Runner.load() applies the PPO training seed. Restore the
            # evaluation seed after every model is loaded and reset the board.
            random.seed(job["seed"])
            np.random.seed(job["seed"])
            torch.manual_seed(job["seed"])
            torch.cuda.manual_seed_all(job["seed"])
            observations, _ = env.reset(seed=job["seed"])
            tracker = EvalMetricsTracker(env)
            previous_actions = torch.zeros(num_envs, 4, device=device)
            is_first = torch.ones(num_envs, dtype=torch.bool, device=device)
            last_saved = 0
            with tqdm(total=job["games"], desc=job["id"], mininterval=5) as progress:
                while tracker.total_games < job["games"] and simulation_app.is_running():
                    actions = torch.cat(
                        [agent.action(observations, is_first, previous_actions) for agent in agents],
                        dim=-1,
                    )
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Policy produced non-finite actions")
                    # Save policy-frame actions before any in-place frame/scaling wrappers.
                    previous_actions = actions.clone()
                    observations, _, terminated, truncated, _ = env.step(actions)
                    is_first = (terminated | truncated).bool()
                    # Use post-reset obs directly. Never act on a previous game's
                    # terminal image or carry its RSSM state into the next game.
                    tracker.update()
                    accepted = is_first & (completed < quotas)
                    ids = accepted.nonzero(as_tuple=True)[0]
                    if len(ids):
                        tracker.finalize(accepted)
                        env_ids.extend(ids.cpu().tolist())
                        episode_ids.extend(completed[ids].cpu().tolist())
                        completed[ids] += 1
                        progress.update(len(ids))
                        progress.set_postfix(**tracker.live_postfix(), refresh=False)
                    if tracker.total_games - last_saved >= 1000:
                        save()
                        last_saved = tracker.total_games
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving completed games.", flush=True)
    finally:
        try:
            save()
        finally:
            env.close()
    if tracker is not None:
        print("\n".join(tracker.summary_body_lines()))
    return 0 if not interrupted and tracker is not None and tracker.total_games == job["games"] else 130


if __name__ == "__main__":
    raise SystemExit(main())

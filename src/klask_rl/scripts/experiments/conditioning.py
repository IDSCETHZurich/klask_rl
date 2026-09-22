"""Evaluate a frozen D-RSSM on logged images/actions, without Isaac Lab."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "tournament"))
sys.path.insert(0, str(SCRIPTS / "dreamer"))
from raw_data import utc_now, write_json
from run_tournament import sha256


def conditioning_actions(actions, condition, timing, seed):
    """Commands indexed by departure frame t for prediction of frame t+1."""
    result = actions.clone()
    if timing == "legacy":
        result[1:, 2:] = actions[:-1, 2:]
        result[0, 2:] = 0
    elif timing != "aligned":
        raise ValueError(timing)
    if condition == "zero":
        result[:, 2:] = 0
    elif condition == "random":
        generator = torch.Generator(device=actions.device).manual_seed(seed)
        result[:, 2:] = torch.rand(actions[:, 2:].shape, device=actions.device, generator=generator) * 2 - 1
    elif condition != "exact":
        raise ValueError(condition)
    return result


def raw_kl(posterior_logits, prior_logits):
    """Repository training KL before free-nat clipping or loss weighting (nats)."""
    log_q = posterior_logits.log_softmax(-1)
    log_p = prior_logits.log_softmax(-1)
    return (log_q.exp() * (log_q - log_p)).sum((-1, -2))


def open_loop(agent, initial, actions, embeddings, images):
    """Target images inform diagnostics only, never the open-loop latent rollout."""
    stoch, deter = initial
    for k, action in enumerate(actions):
        deter = agent.rssm._deter_net(stoch, deter, action[None])
        prior_logits = agent.rssm._img_net(deter)
        stoch = agent.rssm.get_dist(prior_logits).rsample()
        posterior_logits = agent.rssm._obs_net(torch.cat((deter, embeddings[k : k + 1]), -1))
        post_stoch = agent.rssm.get_dist(posterior_logits).rsample()
        target = images[k : k + 1, None].float() / 255.0
        prior_image = agent.decoder(stoch[:, None], deter[:, None])["image"]
        post_image = agent.decoder(post_stoch[:, None], deter[:, None])["image"]
        nll = -prior_image.log_prob(target).item()
        yield {
            "horizon": k + 1,
            "image_nll": nll,
            "image_nll_per_pixel": nll / images[k].numel(),
            "posterior_image_nll": -post_image.log_prob(target).item(),
            "kl_nats": raw_kl(posterior_logits, prior_logits).item(),
        }


def evaluate_trajectory(agent, images, actions, *, burn_in, stride, horizons, draws, seed, timings):
    device = actions.device
    embeddings = torch.cat([agent.encoder({"image": chunk.float() / 255.0}) for chunk in images.split(32)])
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    for timing in timings:
        exact_actions = conditioning_actions(actions, "exact", timing, seed)
        # Filter the shared, exact history once per timing. No conditions are
        # allowed to change the starting posterior or the recorded own commands.
        torch.manual_seed(seed)
        stoch, deter = agent.rssm.initial(1)
        for t in range(len(images) - horizons):
            previous = torch.zeros_like(actions[:1]) if t == 0 else exact_actions[t - 1 : t]
            stoch, deter, _ = agent.rssm.obs_step(
                stoch,
                deter,
                previous,
                embeddings[t : t + 1],
                torch.tensor([t == 0], device=device),
            )
            if t < burn_in or (t - burn_in) % stride:
                continue
            for draw in range(draws):
                sample_seed = seed + t * 1009 + draw * 1000003
                for condition in ("exact", "random", "zero"):
                    commands = conditioning_actions(actions, condition, timing, sample_seed + 17)
                    # Paired Monte Carlo noise without perturbing teacher filtering.
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(sample_seed)
                        rows = open_loop(
                            agent,
                            (stoch.clone(), deter.clone()),
                            commands[t : t + horizons],
                            embeddings[t + 1 : t + 1 + horizons],
                            images[t + 1 : t + 1 + horizons],
                        )
                        for row in rows:
                            yield dict(
                                row,
                                origin=t,
                                condition=condition,
                                timing=timing,
                                draw=draw,
                                sample_seed=sample_seed,
                                target_t=t + row["horizon"],
                            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Tournament directory with logged trajectories",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Relocated copy of the SAME checkpoint; SHA256 must match dataset",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--horizons", type=int, default=15)
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument(
        "--timings",
        nargs="+",
        choices=("aligned", "legacy"),
        default=["aligned", "legacy"],
    )
    args = parser.parse_args()
    if min(args.stride, args.horizons, args.draws) < 1 or args.burn_in < 0:
        parser.error("Need positive stride, horizons and draws, and nonnegative burn-in")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory; diagnostics never overwrite raw results")
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = manifest["agents"]["D-RSSM"]
    if args.checkpoint is not None:
        spec = dict(spec, checkpoint=str(args.checkpoint.resolve()))
    if sha256(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise ValueError("D-RSSM checkpoint changed since trajectory collection")
    indices = sorted(args.dataset.glob("*/trajectories/observations.csv"))
    if not indices:
        raise ValueError("No held-out trajectories found; run the collection experiment first")
    collection_jobs = [job for job in manifest["jobs"] if "trajectory_domain" in job]
    if not collection_jobs:
        raise ValueError("Dataset manifest does not identify held-out trajectory jobs")
    for job in collection_jobs:
        leg = args.dataset / job["id"]
        if not (leg / "metadata.json").exists() or not (leg / "trajectories/observations.csv").exists():
            raise ValueError(f"Collection is incomplete: {job['id']}")
        info = json.loads((leg / "metadata.json").read_text())
        if not info["complete"] or info.get("recorded_trajectory_games") != job["games"]:
            raise ValueError(f"Finish the fixed-quota collection before analysing: {job['id']}")
    import gymnasium as gym
    from checkpoint_loaders import _load_dreamer_agent
    from worker import configuration_for_job

    with indices[0].open() as stream:
        first_index = next(csv.DictReader(stream))
    with np.load(indices[0].parent / first_index["shard"], allow_pickle=False) as first:
        image_shape = first["image"].shape[1:]
    cfg = configuration_for_job(manifest, {"device": args.device, "seed": args.seed})
    step_dt = float(cfg.env.sim_dt) * int(cfg.env.decimation)
    spaces = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, image_shape, dtype=np.uint8),
            "policy": gym.spaces.Box(-np.inf, np.inf, (20,), dtype=np.float32),
        }
    )
    model = _load_dreamer_agent(
        cfg,
        spaces,
        gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
        spec["checkpoint"],
        args.device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "started_at": utc_now(),
        "argv": sys.argv,
        "arguments": vars(args),
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_run_id": manifest["run_id"],
        "config": manifest["evaluation_config"],
        "seed": args.seed,
        "complete": False,
        "control_step_seconds": step_dt,
        "dataset_experiment": manifest.get("experiment", {}),
        "environment_steps": 0,
        "logged_observations": 0,
        "rows": 0,
        "trajectory_sha256": {},
        "image_nll_definition": "-decoder.log_prob(image/255); MSEDist sum squared error, not calibrated likelihood",
        "kl_definition": "sum KL(q(z|imagined h,target image)||p(z|imagined h)); raw logits, no free-nat clipping",
        "protocol": "shared exact-filtered burn-in; recorded own commands; no future observation feedback; paired RNG",
        "independence": "horizons, conditions, draws and origins are paired/dependent; bootstrap by whole game",
    }
    write_json(args.output / "metadata.json", metadata)
    fields = [
        "dataset_run_id",
        "job_id",
        "trajectory_id",
        "domain",
        "opponent",
        "seat",
        "checkpoint",
        "seed",
        "origin",
        "target_t",
        "timing",
        "condition",
        "draw",
        "sample_seed",
        "horizon",
        "horizon_seconds",
        "image_nll",
        "image_nll_per_pixel",
        "posterior_image_nll",
        "kl_nats",
    ]
    try:
        with (
            (args.output / "conditioning.csv").open("w", newline="") as stream,
            torch.inference_mode(),
        ):
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for index_path in indices:
                job_id = index_path.parent.parent.name
                job = next(job for job in manifest["jobs"] if job["id"] == job_id)
                job_metadata = json.loads((index_path.parent.parent / "metadata.json").read_text())
                metadata["environment_steps"] += job_metadata["environment_steps"]
                with index_path.open() as index_stream:
                    index = list(csv.DictReader(index_stream))
                trajectories = {row["trajectory_id"]: row for row in index}
                for trajectory, row in trajectories.items():
                    trajectory_seed = int.from_bytes(
                        hashlib.sha256(f"{args.seed}:{job_id}:{trajectory}".encode()).digest()[:4],
                        "little",
                    )
                    shard = index_path.parent / row["shard"]
                    metadata["trajectory_sha256"][str(shard.resolve())] = sha256(shard)
                    with np.load(shard, allow_pickle=False) as data:
                        images = torch.as_tensor(data["image"], device=args.device)
                        actions = torch.as_tensor(data["action"], device=args.device)
                    metadata["logged_observations"] += len(images)
                    if len(images) <= args.burn_in + args.horizons:
                        continue
                    prefix = {
                        "dataset_run_id": manifest["run_id"],
                        "job_id": job_id,
                        "trajectory_id": trajectory,
                        "domain": row["domain"],
                        "opponent": job["seat1" if row["seat"] == "0" else "seat0"],
                        "seat": row["seat"],
                        "checkpoint": spec["checkpoint"],
                        "seed": args.seed,
                    }
                    for result in evaluate_trajectory(
                        model,
                        images,
                        actions,
                        burn_in=args.burn_in,
                        stride=args.stride,
                        horizons=args.horizons,
                        draws=args.draws,
                        seed=trajectory_seed,
                        timings=args.timings,
                    ):
                        result["horizon_seconds"] = result["horizon"] * step_dt
                        writer.writerow(dict(prefix, **result))
                        metadata["rows"] += 1
                    stream.flush()
                    print(f"{job_id}/{trajectory}: {metadata['rows']} rows", flush=True)
        metadata["complete"] = metadata["rows"] > 0
    finally:
        metadata["updated_at"] = utc_now()
        write_json(args.output / "metadata.json", metadata)
    if not metadata["complete"]:
        raise SystemExit("No eligible windows; collect longer games or reduce burn-in")


if __name__ == "__main__":
    main()

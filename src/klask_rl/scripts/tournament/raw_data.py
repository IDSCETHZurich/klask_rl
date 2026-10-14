"""Portable, atomic raw exports. CSV is the public analysis interface."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def export_games(npz_path, job, run_id):
    """One row per game, explicitly in seat-0 perspective; retain every raw flag."""
    with np.load(npz_path, allow_pickle=False) as data:
        n = int(data["num_games"])
        if int(data["outcome_schema_version"]) != 2:
            raise ValueError("Cannot export ambiguous legacy outcome codes")
        arrays = {
            key: data[key]
            for key in data.files
            if data[key].ndim == 1
            and len(data[key]) == n
            and key not in ("termination_names", "outcome_legend")
            and not key.startswith("meta_")
        }
        names = data["termination_names"].tolist()
        fields = [
            "run_id",
            "job_id",
            "match",
            "seat0",
            "seat1",
            "condition",
            "seed",
            "game_id",
            "payoff_seat0",
        ]
        fields += list(arrays) + ["term_" + name for name in names]
        rows = []
        for i in range(n):
            code = int(data["outcome"][i])
            if code not in range(5):
                raise ValueError(f"Invalid outcome {code}")
            row = {key: job[key] for key in ("match", "seat0", "seat1", "condition", "seed")}
            row.update(
                run_id=run_id,
                job_id=job["id"],
                game_id=i,
                payoff_seat0=1.0 if code in (0, 3) else 0.5 if code == 4 else 0.0,
            )
            row.update({key: values[i].item() for key, values in arrays.items()})
            row.update({"term_" + name: int(data["termination_flags"][i, j]) for j, name in enumerate(names)})
            rows.append(row)
    write_csv(Path(npz_path).with_suffix(".csv"), rows, fields)


class TrajectoryRecorder:
    """Completed held-out games: indexed CSV actions plus lossless uint8 image shards.

    Images and commands are captured BEFORE stepping. No reset image is ever used
    as a terminal target. The last command has no successor image and is excluded
    from offline target windows. Only quota-accepted games are committed.
    """

    fields = (
        "trajectory_id",
        "domain",
        "seat",
        "env_id",
        "episode_id",
        "t",
        "shard",
        "image_index",
        "own_x",
        "own_y",
        "opponent_x",
        "opponent_y",
    )

    def __init__(self, directory, num_envs, seat, domain):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        self.seat, self.domain = seat, domain
        self.buffers = [[] for _ in range(num_envs)]
        self.rows = []
        self.completed_games = 0

    def capture(self, observations, actions, active):
        key = "image" if self.seat == 0 else "opponent_image"
        ids = active.nonzero(as_tuple=True)[0].cpu().tolist()
        images = observations[key][ids].detach().cpu().numpy()
        commands = actions[ids].detach().cpu().numpy()
        if images.dtype != np.uint8:
            raise ValueError("Trajectory images must retain the renderer's uint8 representation")
        order = [0, 1, 2, 3] if self.seat == 0 else [2, 3, 0, 1]
        for index, env_id in enumerate(ids):
            self.buffers[env_id].append((images[index].copy(), commands[index, order].copy()))

    def finish(self, env_id, episode_id):
        frames = self.buffers[env_id]
        self.buffers[env_id] = []
        trajectory = f"env{env_id:04d}_episode{episode_id:06d}"
        shard = trajectory + ".npz"
        images = np.stack([frame[0] for frame in frames])
        actions = np.stack([frame[1] for frame in frames])
        temp = self.directory / (trajectory + ".tmp.npz")
        np.savez_compressed(temp, image=images, action=actions, schema_version=1)
        temp.replace(self.directory / shard)
        for t, action in enumerate(actions):
            self.rows.append(
                dict(
                    zip(
                        self.fields,
                        [
                            trajectory,
                            self.domain,
                            self.seat,
                            env_id,
                            episode_id,
                            t,
                            shard,
                            t,
                            *action.tolist(),
                        ],
                    )
                )
            )
        write_csv(self.directory / "observations.csv", self.rows, self.fields)
        self.completed_games += 1

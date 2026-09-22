"""CSV/checkpoint instrumentation shared by the existing baseline trainers."""

import csv
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "tournament"))
from raw_data import utc_now, write_csv, write_json
from run_tournament import sha256

DEFAULT_TARGETS = (
    1_000_000,
    3_000_000,
    6_000_000,
    9_000_000,
    15_000_000,
    30_000_000,
    60_000_000,
    100_000_000,
    200_000_000,
    400_000_000,
    800_000_000,
    1_600_000_000,
    3_200_000_000,
    6_400_000_000,
    12_800_000_000,
    25_600_000_000,
)


class TrainingRecorder:
    """Publish checkpoints atomically, then add them to the CSV evaluation queue."""

    checkpoint_fields = (
        "algorithm",
        "seed",
        "environment_steps",
        "iteration",
        "requested_steps",
        "checkpoint",
        "sha256",
        "timestamp",
        "elapsed_seconds",
        "final",
    )

    def __init__(
        self,
        directory,
        algorithm,
        seed,
        config,
        *,
        targets=DEFAULT_TARGETS,
        hours=36,
        argv=None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "metadata.json").exists():
            raise ValueError("Training output already exists; use a new seed directory")
        (self.directory / "checkpoints").mkdir(exist_ok=True)
        self.algorithm, self.seed = algorithm, seed
        self.pending = sorted(set(targets))
        self.rows = []
        self.started = time.monotonic()
        self.hours = hours
        self.last_steps = self.last_iteration = 0
        self.metadata = {
            "schema_version": 1,
            "algorithm": algorithm,
            "seed": seed,
            "config": config,
            "started_at": utc_now(),
            "argv": sys.argv if argv is None else argv,
            "initialization": "scratch",
            "training_opponent": "algorithm-specific self-play",
            "step_unit": "individual environment transitions, summed across environments",
            "requested_checkpoint_steps": self.pending.copy(),
            "wall_hours_per_seed": hours,
            "complete": False,
            "environment_steps": 0,
        }
        write_json(self.directory / "metadata.json", self.metadata)
        self.stream = (self.directory / "training.csv").open("w", newline="")
        self.writer = csv.DictWriter(
            self.stream,
            [
                "algorithm",
                "seed",
                "environment_steps",
                "iteration",
                "elapsed_seconds",
                "mean_training_reward",
            ],
        )
        self.writer.writeheader()

    def save(self, steps, iteration, save_function, *, final=False):
        due = [target for target in self.pending if target <= steps]
        if not due and not final:
            return
        if self.rows and self.rows[-1]["environment_steps"] == steps:
            self.rows[-1]["final"] |= final
        else:
            suffix = ".pth" if self.algorithm == "ppo" else ".pt"
            path = self.directory / "checkpoints" / f"step_{steps:012d}{suffix}"
            temporary = path.with_name(path.stem + ".tmp" + suffix)
            save_function(temporary)
            temporary.replace(path)
            self.rows.append(
                {
                    "algorithm": self.algorithm,
                    "seed": self.seed,
                    "environment_steps": steps,
                    "iteration": iteration,
                    "requested_steps": ";".join(map(str, due)),
                    "checkpoint": str(path.resolve()),
                    "sha256": sha256(path),
                    "timestamp": utc_now(),
                    "elapsed_seconds": time.monotonic() - self.started,
                    "final": final,
                }
            )
        self.pending = [target for target in self.pending if target > steps]
        write_csv(self.directory / "checkpoints.csv", self.rows, self.checkpoint_fields)

    def tick(self, steps, iteration, save_function, reward=None):
        if steps <= self.last_steps:
            return False
        self.last_steps, self.last_iteration = steps, iteration
        self.writer.writerow(
            {
                "algorithm": self.algorithm,
                "seed": self.seed,
                "environment_steps": steps,
                "iteration": iteration,
                "elapsed_seconds": time.monotonic() - self.started,
                "mean_training_reward": reward,
            }
        )
        self.stream.flush()
        self.save(steps, iteration, save_function)
        self.metadata.update(environment_steps=steps, iteration=iteration, updated_at=utc_now())
        write_json(self.directory / "metadata.json", self.metadata)
        return self.hours > 0 and time.monotonic() - self.started >= self.hours * 3600

    def finish(self, save_function, complete):
        try:
            # An exception may interrupt an optimizer update. Keep the last
            # atomic, fully updated checkpoint instead of mislabelling that state.
            if self.last_steps and complete:
                self.save(self.last_steps, self.last_iteration, save_function, final=True)
            self.metadata.update(
                complete=complete,
                finished_at=utc_now(),
                elapsed_seconds=time.monotonic() - self.started,
            )
            write_json(self.directory / "metadata.json", self.metadata)
        finally:
            self.stream.close()


class PPOExperimentObserver:
    """Delegate existing training behavior; add post-update snapshots and counters."""

    def __init__(self, observer, recorder):
        self.observer, self.recorder = observer, recorder
        self.algo = None

    def __getattr__(self, name):
        return getattr(self.observer, name)

    def after_init(self, algo):
        self.algo = algo
        self.observer.after_init(algo)

    def save(self, path):
        self.algo.save(str(path.with_suffix("")))  # rl_games appends .pth

    def after_print_stats(self, frame, epoch_num, total_time):
        self.observer.after_print_stats(frame, epoch_num, total_time)
        reward = None
        if self.algo.game_rewards.current_size:
            reward = float(self.algo.game_rewards.get_mean()[0])
        if self.recorder.tick(int(frame), int(epoch_num), self.save, reward):
            # rl_games checks this limit immediately after observer callbacks.
            self.algo.max_epochs = epoch_num

    def finish(self, complete):
        self.recorder.finish(self.save, complete)

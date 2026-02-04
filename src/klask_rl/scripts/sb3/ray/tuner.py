# Copyright (c) 2024-2026, The Klask RL Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ray Tune integration for Klask RL hyperparameter tuning.

This script follows the Isaac Lab Ray workflow pattern.

Usage:
    # Local tuning
    python scripts/sb3/ray/tuner.py --run_mode local \
        --cfg_file scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py \
        --cfg_class KlaskSacHerTuneJobCfg

Examples:
    python scripts/sb3/ray/tuner.py --run_mode local \
        --cfg_file scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py \
        --cfg_class KlaskSacHerTuneJobCfg \
        --num_samples 8
"""

import argparse
import importlib.util
import os
from pathlib import Path
from time import sleep

import ray
import util
from ray import air, tune
from ray.tune.search.optuna import OptunaSearch

BASE_DIR = os.path.expanduser("~")
PYTHON_EXEC = "python"
WORKFLOW = "scripts/sb3/train_sac.py"
PROCESS_RESPONSE_TIMEOUT = 200.0
MAX_LINES_TO_SEARCH_EXPERIMENT_LOGS = 1000
MAX_LOG_EXTRACTION_ERRORS = 2


class KlaskRLTuneTrainable(tune.Trainable):
    """The Klask RL Ray Tune Trainable.

    This class uses the SB3 training workflow with Hydra integration.
    Achieves Ray-based logging by reading tensorboard logs from the training runs.
    """

    def setup(self, config: dict) -> None:
        """Get the invocation command, return quick for easy scheduling."""
        self.data = None
        self.time_since_last_proc_response = 0.0
        self.invoke_cmd = util.get_invocation_command_from_cfg(cfg=config, python_cmd=PYTHON_EXEC, workflow=WORKFLOW)
        print(f"[INFO]: Recovered invocation with {self.invoke_cmd}")
        self.experiment = None

    def reset_config(self, new_config: dict):
        """Allow environments to be re-used by fetching a new invocation command."""
        self.setup(new_config)
        return True

    def step(self) -> dict:
        if self.experiment is None:  # start experiment
            print(f"[INFO]: Invoking experiment as first step with {self.invoke_cmd}...")
            try:
                experiment = util.execute_job(
                    self.invoke_cmd,
                    identifier_string="",
                    extract_experiment=True,
                    persistent_dir=BASE_DIR,
                    max_lines_to_search_logs=MAX_LINES_TO_SEARCH_EXPERIMENT_LOGS,
                    max_time_to_search_logs=PROCESS_RESPONSE_TIMEOUT,
                )
                self.experiment = experiment
                print(f"[INFO]: Started experiment {self.experiment}")
            except util.LogExtractionError as e:
                print(f"[ERROR]: Could not start experiment: {e}")
                return {"done": True, "error": str(e)}

        # Load latest metrics from tensorboard
        try:
            log_dir = self.experiment["log_dir"]
            # Find the latest run directory
            experiment_name = self.experiment["experiment_name"]
            full_log_dir = os.path.join(log_dir, experiment_name)

            if os.path.exists(full_log_dir):
                scalars = util.load_tensorboard_logs(full_log_dir)
                if scalars:
                    self.data = scalars
                    self.time_since_last_proc_response = 0.0
                else:
                    self.time_since_last_proc_response += 1.0
            else:
                self.time_since_last_proc_response += 1.0

            # Check if process has stopped responding
            if self.time_since_last_proc_response > PROCESS_RESPONSE_TIMEOUT:
                print(f"[WARNING]: Process stopped responding after {PROCESS_RESPONSE_TIMEOUT}s")
                return {"done": True, "timeout": True}

            sleep(1.0)  # Don't hammer the filesystem

            if self.data:
                return self.data
            else:
                return {}

        except Exception as e:
            print(f"[ERROR]: Exception while loading logs: {e}")
            return {"done": True, "error": str(e)}

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Save checkpoint (not implemented for now)."""
        pass

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load checkpoint (not implemented for now)."""
        pass


def main():
    parser = argparse.ArgumentParser(description="Klask RL Ray Tune")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="local",
        choices=["local"],
        help="Run mode (only local supported for now)",
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Path to config file (e.g., scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py)",
    )
    parser.add_argument(
        "--cfg_class",
        type=str,
        required=True,
        help="Config class name (e.g., KlaskSacHerTuneJobCfg)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of hyperparameter samples to try",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rollout_ep_rew_mean",
        help="Metric to optimize (tensorboard scalar key)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="max",
        choices=["min", "max"],
        help="Optimization mode",
    )
    args = parser.parse_args()

    # Load config class
    cfg_path = Path(args.cfg_file)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path

    spec = importlib.util.spec_from_file_location("config_module", cfg_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    cfg_class = getattr(config_module, args.cfg_class)
    job_cfg = cfg_class()

    # Initialize Ray
    if args.run_mode == "local":
        ray.init(ignore_reinit_error=True)

    # Create search space
    search_space = job_cfg.to_dict()

    # Run tuning
    tuner = tune.Tuner(
        KlaskRLTuneTrainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            metric=args.metric,
            mode=args.mode,
            search_alg=OptunaSearch(),
        ),
        run_config=air.RunConfig(
            name="klask_sac_tune",
            local_dir=os.path.join(BASE_DIR, "ray_results"),
        ),
    )

    results = tuner.fit()
    print(results.get_best_result())


if __name__ == "__main__":
    main()

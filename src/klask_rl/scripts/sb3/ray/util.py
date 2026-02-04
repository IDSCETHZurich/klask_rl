# Copyright (c) 2024-2026, The Klask RL Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for Ray Tune integration with Klask RL."""

import os
import re
import subprocess
from datetime import datetime
from tensorboard.backend.event_processing.directory_watcher import DirectoryDeletedError
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(directory: str) -> dict:
    """From a tensorboard directory, get the latest scalar values.

    Args:
        directory: The directory of the tensorboard logging.

    Returns:
        The latest available scalar values.
    """

    def replace_invalid_chars(t):
        t2 = re.sub(r"[^0-9A-Za-z_./]", "_", t)
        t2 = re.sub(r"_+", "_", t2)
        return t2.strip("_")

    def get_latest_scalars(path: str) -> dict:
        event_acc = EventAccumulator(path, size_guidance={"scalars": 1})
        try:
            event_acc.Reload()
            if event_acc.Tags()["scalars"]:
                return {
                    replace_invalid_chars(tag): event_acc.Scalars(tag)[-1].value
                    for tag in event_acc.Tags()["scalars"]
                    if event_acc.Scalars(tag)
                }
        except (KeyError, OSError, RuntimeError, DirectoryDeletedError):
            return {}

    scalars = get_latest_scalars(directory)
    return scalars or get_latest_scalars(os.path.join(directory, "summaries"))


def get_invocation_command_from_cfg(
    cfg: dict,
    python_cmd: str = "python",
    workflow: str = "scripts/sb3/train_sac.py",
) -> str:
    """Generate command with proper Hydra arguments.

    Args:
        cfg: Config dict with 'runner_args' and 'hydra_args' keys.
        python_cmd: Python executable command.
        workflow: Path to the training script.

    Returns:
        Full invocation command string.
    """
    runner_args = []
    hydra_args = []

    def process_args(args, target_list, is_hydra=False):
        for key, value in args.items():
            if not is_hydra:
                # Runner args (--config, etc.)
                if key.startswith("--"):
                    target_list.append(f"{key} {value}")
                else:
                    target_list.append(f"{value}")
            else:
                # Hydra args (agent.learning_rate=0.0003, etc.)
                if isinstance(value, list):
                    formatted_items = [str(x) for x in value]
                    target_list.append(f"'{key}=[{','.join(formatted_items)}]'")
                elif isinstance(value, str) and ("{" in value or "}" in value):
                    target_list.append(f"'{key}={value}'")
                else:
                    target_list.append(f"{key}={value}")

    print(f"[INFO]: Starting workflow {workflow}")
    process_args(cfg["runner_args"], runner_args)
    print(f"[INFO]: Retrieved workflow runner args: {runner_args}")
    process_args(cfg["hydra_args"], hydra_args, is_hydra=True)
    print(f"[INFO]: Retrieved hydra args: {hydra_args}")

    invoke_cmd = f"{python_cmd} {workflow} "
    invoke_cmd += " ".join(runner_args) + " " + " ".join(hydra_args)
    return invoke_cmd


class LogExtractionError(Exception):
    """Raised when we cannot extract experiment_name/logdir from the trainer output."""

    pass


def execute_job(
    job_cmd: str,
    identifier_string: str = "job 0",
    extract_experiment: bool = False,
    persistent_dir: str | None = None,
    max_lines_to_search_logs: int = 1000,
    max_time_to_search_logs: float = 200.0,
) -> str | dict:
    """Issue a job (shell command).

    Args:
        job_cmd: The shell command to run.
        identifier_string: What prefix to add to make logs easier to differentiate.
        extract_experiment: When true, search for experiment details from a training run.
        persistent_dir: When supplied, change to run the directory in a persistent directory.
        max_lines_to_search_logs: Maximum number of lines to search for experiment info.
        max_time_to_search_logs: Maximum time to wait for experiment info before giving up.

    Returns:
        Relevant information from the job.
    """
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    print(f"[{identifier_string}] [{start_time}] {job_cmd}")

    if persistent_dir:
        os.chdir(persistent_dir)

    process = subprocess.Popen(
        job_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    experiment_name = None
    log_dir = None
    lines_searched = 0

    if extract_experiment:
        # Search for experiment name and log directory in the output
        for line in process.stdout:
            print(f"[{identifier_string}] {line.rstrip()}")
            lines_searched += 1

            if "Experiment name:" in line:
                experiment_name = line.split("Experiment name:")[-1].strip()
            if "Logging experiment in directory:" in line:
                log_dir = line.split("Logging experiment in directory:")[-1].strip()

            if experiment_name and log_dir:
                break

            if lines_searched >= max_lines_to_search_logs:
                raise LogExtractionError(f"Could not find experiment info in first {max_lines_to_search_logs} lines")

        if not (experiment_name and log_dir):
            raise LogExtractionError("Could not extract experiment name and log directory")

        # Continue consuming output to prevent blocking
        for line in process.stdout:
            print(f"[{identifier_string}] {line.rstrip()}")

        process.wait()
        return {"experiment_name": experiment_name, "log_dir": log_dir}
    else:
        for line in process.stdout:
            print(f"[{identifier_string}] {line.rstrip()}")

        process.wait()
        return "Job completed"

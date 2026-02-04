# Copyright (c) 2024-2026, The Klask RL Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hyperparameter tuning configuration for Klask SAC training.

This follows the Isaac Lab Ray integration pattern with runner_args and hydra_args.
"""

import pathlib
import sys

# Allow for import of items from the ray workflow
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent
sys.path.extend([str(UTIL_DIR), str(CUR_DIR)])

from ray import tune


class KlaskSacBaseJobCfg:
    """Base SAC job config for Klask environment."""

    def __init__(self, cfg: dict = {}):
        """Initialize the job config.

        Args:
            cfg: Optional config dict to merge with defaults.
        """
        # Runner arguments (standard CLI flags)
        self.runner_args = {
            "--config": "experiments/klask_sac_base.yaml",
        }

        # Hydra arguments (config overrides)
        self.hydra_args = {}

        # Merge with provided config
        if "runner_args" in cfg:
            self.runner_args.update(cfg["runner_args"])
        if "hydra_args" in cfg:
            self.hydra_args.update(cfg["hydra_args"])

    def to_dict(self) -> dict:
        """Convert to dict for Ray Tune."""
        return {
            "runner_args": self.runner_args,
            "hydra_args": self.hydra_args,
        }


class KlaskSacHerJobCfg:
    """HER SAC job config for Klask environment."""

    def __init__(self, cfg: dict = {}):
        self.runner_args = {
            "--config": "experiments/klask_sac_her.yaml",
        }

        self.hydra_args = {}

        if "runner_args" in cfg:
            self.runner_args.update(cfg["runner_args"])
        if "hydra_args" in cfg:
            self.hydra_args.update(cfg["hydra_args"])

    def to_dict(self) -> dict:
        return {
            "runner_args": self.runner_args,
            "hydra_args": self.hydra_args,
        }


class KlaskSacHerTuneJobCfg(KlaskSacHerJobCfg):
    """HER SAC job with hyperparameter tuning for Klask environment."""

    def __init__(self, cfg: dict = {}):
        super().__init__(cfg)

        # Add hyperparameter search space
        self.hydra_args.update(
            {
                "agent.learning_rate": tune.loguniform(1e-5, 3e-4),
                "agent.batch_size": tune.choice([256, 512, 1024]),
                "agent.gamma": tune.uniform(0.95, 0.995),
                "agent.tau": tune.loguniform(1e-3, 2e-2),
            }
        )


class KlaskSacTwoStageHerJobCfg:
    """Two-stage HER SAC job config for Klask environment."""

    def __init__(self, cfg: dict = {}):
        self.runner_args = {
            "--config": "experiments/klask_sac_two_stage_her.yaml",
        }

        self.hydra_args = {}

        if "runner_args" in cfg:
            self.runner_args.update(cfg["runner_args"])
        if "hydra_args" in cfg:
            self.hydra_args.update(cfg["hydra_args"])

    def to_dict(self) -> dict:
        return {
            "runner_args": self.runner_args,
            "hydra_args": self.hydra_args,
        }

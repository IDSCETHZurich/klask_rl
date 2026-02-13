# Copyright (c) 2024-2026, The Klask RL Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hyperparameter tuning configuration for Klask SAC training.

This follows the Isaac Lab Ray integration pattern. Each config class specifies:
- runner_args: CLI arguments like --config (the base YAML config file)
- hydra_args: Hydra overrides on top of that base config for hyperparameter tuning

For usage with Isaac Lab's tuner:
    cd /workspace/isaaclab
    ./isaaclab.sh -p scripts/reinforcement_learning/ray/tuner.py --run_mode local \\
        --cfg_file /workspace/klask_rl/scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py \\
        --cfg_class KlaskSacTwoStageHerJobCfg \\
        --num_samples 8 \\
        --workflow /workspace/klask_rl/scripts/sb3/train_sac.py \\
        --metric two_stage_goal_scores_count
    
    Available metrics for two-stage HER (Ray Tune converts / to _):
        - rollout_ep_rew_mean: Average episode reward (default)
        - two_stage_goal_scores_count: Number of envs that scored goals per rollout
        - two_stage_goal_score_rate: Fraction of envs that scored goals
        - two_stage_ball_hits_count: Number of envs that hit the ball
        - two_stage_ball_hit_rate: Fraction of envs that hit the ball
"""

import numpy as np
from ray import tune


class KlaskSacBaseJobCfg:
    """Base SAC job config for Klask environment.

    Loads experiments/klask_sac_base.yaml. Uses single-value choice to satisfy Ray Tune.
    """

    def __init__(self):
        """Initialize the job config.

        Isaac Lab's tuner expects self.cfg as a dict with 'runner_args' and 'hydra_args'.
        - runner_args: CLI arguments passed before Hydra args (--config, --task, etc.)
        - hydra_args: Hydra overrides that will be appended as CLI args (agent.learning_rate=0.0003)
        """
        self.cfg = {
            "runner_args": {
                "--config": "experiments/klask_sac_base.yaml",
            },
            "hydra_args": {
                "agent.learning_rate": tune.choice([3.0e-4]),
            },
        }


class KlaskSacHerJobCfg:
    """HER SAC job config for Klask environment.

    Loads experiments/klask_sac_her.yaml. Uses single-value choices to satisfy
    Ray Tune/OptunaSearch requirements while running deterministically.
    """

    def __init__(self):
        self.cfg = {
            "runner_args": {
                "--config": "experiments/klask_sac_her.yaml",
            },
            "hydra_args": {
                # Single-value choices - satisfies Ray Tune but runs deterministically
                "agent.learning_rate": tune.choice([3.0e-4]),
            },
        }


class KlaskSacHerTuneJobCfg:
    """HER SAC job with hyperparameter tuning for Klask environment.

    Loads experiments/klask_sac_her.yaml as base config, then varies hyperparameters.
    These Hydra overrides will be applied on top of the YAML config values.
    """

    def __init__(self):
        self.cfg = {
            "runner_args": {
                "--config": "experiments/klask_sac_her.yaml",
            },
            "hydra_args": {
                # Tune agent hyperparameters via Hydra overrides
                # These override the values loaded from the YAML file
                # "agent.learning_rate": tune.loguniform(1e-5, 3e-4),
                # "agent.batch_size": tune.choice([256, 512, 1024]),
                # "agent.gamma": tune.uniform(0.95, 0.995),
                # "agent.tau": tune.loguniform(1e-3, 2e-2),
                # "agent.buffer_size": tune.choice([100000, 500000, 1000000]),
                # "agent.learning_starts": tune.choice([1000, 5000, 10000]),
                "agent.policy_kwargs.net_arch": tune.choice([[64, 64], [256, 128, 64]]),
            },
        }

        self.stop = {
            "time_total_s": 86400,  # Stop after 24 hours
        }


class KlaskSacTwoStageHerJobCfg:
    """Two-stage HER SAC job config for Klask environment.

    Loads experiments/klask_sac_two_stage_her.yaml. Uses single-value choice to satisfy Ray Tune.
    """

    def __init__(self):
        self.cfg = {
            "runner_args": {
                "--config": "experiments/klask_sac_two_stage_her.yaml",
            },
            "hydra_args": {
                "agent.learning_rate": tune.loguniform(1e-5, 3e-3),
                "agent.buffer_size": tune.choice([500_000, 1_000_000, 10_000_000]),
                "agent.batch_size": tune.choice([1024, 2048]),
                "agent.train_freq": tune.choice([32, 64, 128]),
                "agent.gradient_steps": tune.choice([32, 64, 128]),
                "agent.policy_kwargs.net_arch": tune.choice(
                    [
                        [256, 128, 64],
                        [256, 256, 128, 64],
                        [512, 256, 128, 64],
                        [256, 128, 64, 64],
                    ]
                ),
                "her.n_sampled_goal": tune.choice([2, 4, 8]),
                "two_stage.ball_hit_timeout": tune.choice([0.5, 1.0, 2.0]),
                "two_stage.ball_hit_env_reward": tune.loguniform(1.0, 1000.0),
                "two_stage.goal_score_env_reward": tune.sample_from(
                    lambda spec: np.exp(
                        np.random.uniform(
                            np.log(spec["hydra_args"]["two_stage.ball_hit_env_reward"]),
                            np.log(10000.0),
                        )
                    )
                ),
                "two_stage.ball_hit_wrapper_reward": tune.loguniform(1.0, 1000.0),
                "two_stage.goal_score_wrapper_reward": tune.sample_from(
                    lambda spec: np.exp(
                        np.random.uniform(
                            np.log(
                                spec["hydra_args"]["two_stage.ball_hit_wrapper_reward"]
                            ),
                            np.log(10000.0),
                        )
                    )
                ),
            },
        }

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


class BallHitRateStopper(tune.Stopper):
    """Stop a trial early if ball_hit_rate hasn't exceeded a threshold after N timesteps.

    Stops individual trials (not the entire experiment). If a trial has trained
    for at least ``min_timesteps`` and the ``two_stage/ball_hit_rate`` metric is
    still below ``min_rate``, the trial is terminated so resources can be used
    for the next sample.

    Usage:
        Injected automatically via ``_inject_stopper()`` when the cfg class is instantiated.
    """

    def __init__(self, min_rate: float = 0.5, min_timesteps: int = 100_000_000):
        self._min_rate = min_rate
        self._min_timesteps = min_timesteps

    def __call__(self, trial_id: str, result: dict) -> bool:
        # Handle both "/" and "_" metric key formats (depends on Isaac Lab version)
        timesteps = result.get("time/total_timesteps", result.get("time_total_timesteps", 0))
        ball_hit_rate = result.get("two_stage/ball_hit_rate", result.get("two_stage_ball_hit_rate", None))

        if timesteps >= self._min_timesteps and ball_hit_rate is not None and ball_hit_rate < self._min_rate:
            print(
                f"[STOPPER] Trial {trial_id}: ball_hit_rate={ball_hit_rate:.3f} < "
                f"{self._min_rate} after {timesteps / 1e6:.0f}M steps. Stopping early."
            )
            return True
        return False

    def stop_all(self) -> bool:
        return False


def _inject_stopper(stopper: tune.Stopper) -> None:
    """Monkey-patch air.RunConfig to inject a trial stopper.

    Since the container's tuner.py doesn't support --stopper, we patch
    RunConfig.__init__ to inject our stopper. This is applied when our
    cfg class is instantiated (before RunConfig is constructed).

    Args:
        stopper: A tune.Stopper instance for early stopping of individual trials.
    """
    from ray import air

    if hasattr(air.RunConfig.__init__, "_klask_patched"):
        return  # Already patched

    _original_init = air.RunConfig.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("stop", stopper)
        _original_init(self, *args, **kwargs)

    _patched_init._klask_patched = True
    air.RunConfig.__init__ = _patched_init


def _inject_points_to_evaluate(points_to_evaluate: list[dict]) -> None:
    """Monkey-patch OptunaSearch to inject initial seed points.

    Since Isaac Lab's tuner.py creates OptunaSearch without points_to_evaluate,
    and we can't modify tuner.py (it's in the base Docker image), we patch
    OptunaSearch.__init__ to inject our initial guesses. This patch is applied
    when our cfg class is instantiated (before OptunaSearch is constructed).

    Args:
        points_to_evaluate: List of dicts mapping flattened param keys to values.
            Keys use '/' separators matching Ray Tune's flattened param_space,
            e.g. "hydra_args/agent.learning_rate". Only include params managed
            by OptunaSearch (tune.choice, tune.loguniform, etc.), NOT
            tune.sample_from params (those are sampled independently).
    """
    from ray.tune.search.optuna import OptunaSearch

    if hasattr(OptunaSearch.__init__, "_klask_patched"):
        return  # Already patched, avoid double-patching

    _original_init = OptunaSearch.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("points_to_evaluate", points_to_evaluate)
        _original_init(self, *args, **kwargs)

    _patched_init._klask_patched = True
    OptunaSearch.__init__ = _patched_init


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
                # "agent.learning_rate": tune.loguniform(1e-5, 3e-3),
                # "agent.buffer_size": tune.choice([500_000, 1_000_000, 10_000_000]),
                # "agent.batch_size": tune.choice([1024, 2048]),
                # "agent.train_freq": tune.choice([32, 64, 128]),
                # "agent.gradient_steps": tune.choice([32, 64, 128]),
                "agent.policy_kwargs.net_arch": tune.choice(
                    [
                        # best
                        [512, 256, 256, 128, 64],
                        # [512, 512, 256, 128, 64],
                        # [1024, 512, 256, 128, 64],
                        # [1024, 512, 256, 128, 64, 32],
                        # done
                        # [256, 128, 64, 128, 256],
                        # [512, 256, 128, 64, 128, 256, 512],
                        # [256, 128, 64, 32],
                        # [512, 256, 128, 64, 32],
                        # [1024, 512, 512, 256, 128, 64],
                    ]
                ),
                # "her.n_sampled_goal": tune.choice([2, 4, 8]),
                # "two_stage.ball_hit_timeout": tune.choice([0.5, 1.0, 2.0]),
                # "two_stage.ball_hit_env_reward": tune.loguniform(1.0, 1000.0),
                "two_stage.goal_score_env_reward": tune.choice([1000.0, 2000.0, 5000.0]),
                # "two_stage.goal_score_env_reward": tune.sample_from(
                #     lambda spec: np.exp(
                #         np.random.uniform(
                #             np.log(spec["hydra_args"]["two_stage.ball_hit_env_reward"]),
                #             np.log(10000.0),
                #         )
                #     )
                # ),
                # "two_stage.ball_hit_wrapper_reward": tune.loguniform(1.0, 1000.0),
                # "two_stage.goal_score_wrapper_reward": tune.choice([2.0, 5.0, 10.0, 50.0, 100.0]),
                # "two_stage.goal_score_wrapper_reward": tune.sample_from(
                #     lambda spec: np.exp(
                #         np.random.uniform(
                #             np.log(
                #                 spec["hydra_args"]["two_stage.ball_hit_wrapper_reward"]
                #             ),
                #             np.log(10000.0),
                #         )
                #     )
                # ),
                #      500, 1000, 2000, 5000
                #   2   x     x     x     x
                #   5   x     x     x     x
                #  10   x     x     x     x
                #  50   x     x     x     x
                # 100   x     x     x     x
            },
        }

        # Inject early stopping: stop trial if ball_hit_rate < 0.5 after 100M steps
        # _inject_stopper(BallHitRateStopper(min_rate=0.5, min_timesteps=100_000_000))

        # Initial seed points for OptunaSearch.
        # _inject_points_to_evaluate(
        #     [
        #         {
        #             "hydra_args/agent.learning_rate": 3e-4,
        #             "hydra_args/agent.buffer_size": 1_000_000,
        #             "hydra_args/agent.batch_size": 1024,
        #             "hydra_args/agent.train_freq": 64,
        #             "hydra_args/agent.gradient_steps": 32,
        #             "hydra_args/agent.policy_kwargs.net_arch": [256, 128, 64],
        #             "hydra_args/her.n_sampled_goal": 4,
        #             "hydra_args/two_stage.ball_hit_timeout": 2.0,
        #             "hydra_args/two_stage.ball_hit_env_reward": 500.0,
        #             "hydra_args/two_stage.ball_hit_wrapper_reward": 1.0,
        #         },
        #     ]
        # )

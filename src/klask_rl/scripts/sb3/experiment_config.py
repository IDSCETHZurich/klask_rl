"""Training configuration helpers for SB3 scripts.

This module provides a unified configuration system that loads all training parameters
from a single YAML file. Configuration sections:
- task: The Gym environment ID
- env: Environment settings (num_envs, seed, max_velocity)
- agent: SAC hyperparameters (learning_rate, buffer_size, policy_kwargs, etc.)
- her: HER settings (presence of this section enables HER)
- two_stage: Two-stage HER settings (presence enables two-stage mode)
- training: Training settings (log_interval, checkpoint, video)
- wandb: Weights & Biases logging settings
- app_launcher: Isaac Sim app launcher settings
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Unified experiment configuration loaded from a single YAML file.

    The presence of optional sections determines behavior:
    - If 'her' section exists → HER is enabled
    - If 'two_stage' section exists → Two-stage HER is enabled
    - If 'training.video' section exists → Video recording is enabled
    """

    # Task
    task: str = "Klask-Rl-SAC-v0"

    # Environment settings
    num_envs: int = 4096
    seed: int = 42
    max_velocity: float = 0.2

    # Agent config (SAC hyperparameters) - stored as dict for flexibility
    agent_cfg: dict[str, Any] = field(default_factory=dict)

    # HER settings (None if HER disabled)
    her_cfg: dict[str, Any] | None = None

    # Two-stage HER settings (None if two-stage disabled)
    two_stage_cfg: dict[str, Any] | None = None

    # Training settings
    log_interval: int = 1000
    checkpoint: str | None = None
    export_io_descriptors: bool = False

    # Video settings (None if video disabled)
    video_cfg: dict[str, Any] | None = None

    # Wandb settings (None if wandb disabled)
    wandb_cfg: dict[str, Any] | None = None

    # App launcher settings
    app_launcher: dict[str, Any] = field(default_factory=dict)

    # Path to the config file (for logging)
    config_path: Path | None = None

    @property
    def use_her(self) -> bool:
        """HER is enabled if her_cfg section exists and has content."""
        return self.her_cfg is not None and len(self.her_cfg) > 0

    @property
    def use_two_stage_her(self) -> bool:
        """Two-stage HER is enabled if two_stage_cfg section exists."""
        return self.two_stage_cfg is not None and len(self.two_stage_cfg) > 0

    @property
    def video(self) -> bool:
        """Video recording is enabled if video_cfg section exists."""
        return self.video_cfg is not None and len(self.video_cfg) > 0

    @property
    def video_length(self) -> int:
        """Video length in steps."""
        return self.video_cfg.get("length", 200) if self.video_cfg else 200

    @property
    def video_interval(self) -> int:
        """Interval between video recordings."""
        return self.video_cfg.get("interval", 2000) if self.video_cfg else 2000

    # HER property accessors
    @property
    def her_goal_selection_strategy(self) -> str:
        return (
            self.her_cfg.get("goal_selection_strategy", "future")
            if self.her_cfg
            else "future"
        )

    @property
    def her_n_sampled_goal(self) -> int:
        return self.her_cfg.get("n_sampled_goal", 4) if self.her_cfg else 4

    @property
    def her_achieved_goal_indices(self) -> list[int] | None:
        return self.her_cfg.get("achieved_goal_indices") if self.her_cfg else None

    @property
    def her_desired_goal_indices(self) -> list[int] | None:
        return self.her_cfg.get("desired_goal_indices") if self.her_cfg else None

    @property
    def her_distance_threshold(self) -> float:
        return self.her_cfg.get("distance_threshold", 0.02) if self.her_cfg else 0.02

    @property
    def her_env_reward_scale(self) -> float | None:
        return self.her_cfg.get("env_reward_scale") if self.her_cfg else None

    @property
    def her_wrapper_reward_scale(self) -> float | None:
        return self.her_cfg.get("wrapper_reward_scale") if self.her_cfg else None

    # Two-stage HER property accessors
    @property
    def two_stage_player_pos_indices(self) -> list[int] | None:
        return (
            self.two_stage_cfg.get("player_pos_indices") if self.two_stage_cfg else None
        )

    @property
    def two_stage_ball_pos_indices(self) -> list[int] | None:
        return (
            self.two_stage_cfg.get("ball_pos_indices") if self.two_stage_cfg else None
        )

    @property
    def two_stage_goal_pos_indices(self) -> list[int] | None:
        return (
            self.two_stage_cfg.get("goal_pos_indices") if self.two_stage_cfg else None
        )

    @property
    def two_stage_ball_hit_threshold(self) -> float:
        return (
            self.two_stage_cfg.get("ball_hit_threshold", 0.017)
            if self.two_stage_cfg
            else 0.017
        )

    @property
    def two_stage_goal_score_threshold(self) -> float:
        return (
            self.two_stage_cfg.get("goal_score_threshold", 0.025)
            if self.two_stage_cfg
            else 0.025
        )

    @property
    def two_stage_ball_hit_env_reward(self) -> float | None:
        return (
            self.two_stage_cfg.get("ball_hit_env_reward")
            if self.two_stage_cfg
            else None
        )

    @property
    def two_stage_goal_score_env_reward(self) -> float | None:
        return (
            self.two_stage_cfg.get("goal_score_env_reward")
            if self.two_stage_cfg
            else None
        )

    @property
    def two_stage_ball_hit_wrapper_reward(self) -> float | None:
        return (
            self.two_stage_cfg.get("ball_hit_wrapper_reward")
            if self.two_stage_cfg
            else None
        )

    @property
    def two_stage_goal_score_wrapper_reward(self) -> float | None:
        return (
            self.two_stage_cfg.get("goal_score_wrapper_reward")
            if self.two_stage_cfg
            else None
        )

    # Wandb property accessors
    @property
    def wandb_project(self) -> str | None:
        return self.wandb_cfg.get("project") if self.wandb_cfg else None

    @property
    def wandb_entity(self) -> str | None:
        return self.wandb_cfg.get("entity") if self.wandb_cfg else None

    @property
    def wandb_name(self) -> str | None:
        return self.wandb_cfg.get("name") if self.wandb_cfg else None

    @staticmethod
    def _coerce_int(value: Any, field_name: str) -> int:
        """Convert value to int with proper error handling."""
        if isinstance(value, bool):
            print(f"[ERROR] Config '{field_name}' must be an integer, got bool.")
            sys.exit(1)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return int(float(value))
                except ValueError:
                    print(
                        f"[ERROR] Config '{field_name}' must be numeric, got: {value!r}"
                    )
                    sys.exit(1)
        print(
            f"[ERROR] Config '{field_name}' must be numeric, got: {type(value).__name__}"
        )
        sys.exit(1)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load YAML file and return as dict."""
        if not path.is_file():
            print(f"[ERROR] Config file not found: {path}")
            sys.exit(1)
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            print(f"[ERROR] Failed to load config: {path}\n{exc}")
            sys.exit(1)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            print(f"[ERROR] Config must be a mapping, got: {type(data).__name__}")
            sys.exit(1)
        return data

    @classmethod
    def from_file(cls, path: Path) -> "ExperimentConfig":
        """Load experiment configuration from a unified YAML file."""
        data = cls._load_yaml(path)

        # Extract task
        task = data.get("task", "Klask-Rl-SAC-v0")

        # Extract environment settings
        env_cfg = data.get("env", {})
        num_envs = env_cfg.get("num_envs", 4096)
        seed = env_cfg.get("seed", 42)
        max_velocity = env_cfg.get("max_velocity", 0.2)

        # Extract agent config (SAC hyperparameters)
        agent_cfg = dict(data.get("agent", {}))

        # Extract HER config (None if section doesn't exist or is empty)
        her_cfg = data.get("her")
        if her_cfg is not None and not her_cfg:
            her_cfg = None  # Empty dict means disabled

        # Extract two-stage config
        two_stage_cfg = data.get("two_stage")
        if two_stage_cfg is not None and not two_stage_cfg:
            two_stage_cfg = None

        # Extract training settings
        training_cfg = data.get("training", {})
        log_interval = training_cfg.get("log_interval", 1000)
        checkpoint = training_cfg.get("checkpoint")
        export_io_descriptors = training_cfg.get("export_io_descriptors", False)

        # Extract video config (nested under training)
        video_cfg = training_cfg.get("video")
        if video_cfg is not None and not video_cfg:
            video_cfg = None

        # Extract wandb config
        wandb_cfg = data.get("wandb")
        if wandb_cfg is not None and not wandb_cfg:
            wandb_cfg = None

        # Extract app launcher config
        app_launcher = data.get("app_launcher", {})

        return cls(
            task=task,
            num_envs=num_envs,
            seed=seed,
            max_velocity=max_velocity,
            agent_cfg=agent_cfg,
            her_cfg=her_cfg,
            two_stage_cfg=two_stage_cfg,
            log_interval=log_interval,
            checkpoint=checkpoint,
            export_io_descriptors=export_io_descriptors,
            video_cfg=video_cfg,
            wandb_cfg=wandb_cfg,
            app_launcher=app_launcher,
            config_path=path,
        )

    def setup_cuda_visibility(self) -> None:
        """Set CUDA_VISIBLE_DEVICES based on device in app_launcher config.

        Handles two cases:
        1. Normal training: Sets CUDA_VISIBLE_DEVICES from device (cuda:3 -> CUDA_VISIBLE_DEVICES=3, device=cuda:0)
        2. Ray Tune: CUDA_VISIBLE_DEVICES already set, just remap device to cuda:0

        Must be called BEFORE AppLauncher initialization to prevent Isaac Sim
        from allocating memory on all GPUs.
        """
        import os

        device = self.app_launcher.get("device")
        if device and isinstance(device, str) and device.startswith("cuda:"):
            # Check if Ray Tune (or another process) already set CUDA_VISIBLE_DEVICES
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                # Ray has filtered devices, just remap to logical device 0
                original_device = device
                self.app_launcher["device"] = "cuda:0"
                print(
                    f"[INFO] Ray Tune detected: mapping {original_device} -> cuda:0 "
                    f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
                )
            else:
                # Normal case: set CUDA_VISIBLE_DEVICES ourselves
                try:
                    gpu_id = device.split(":")[1]
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                    print(
                        f"[INFO] Set CUDA_VISIBLE_DEVICES={gpu_id} "
                        f"(Physical GPU {gpu_id} will appear as cuda:0 to the application)"
                    )
                    # Update config to use cuda:0 since we've remapped the GPU
                    self.app_launcher["device"] = "cuda:0"
                except (IndexError, ValueError):
                    print(f"[WARNING] Could not parse GPU ID from device: {device}")

    def app_launcher_args(self) -> dict[str, Any]:
        """Get app launcher arguments, enabling cameras if video is requested."""
        args = dict(self.app_launcher)
        if self.video:
            if args.get("enable_cameras") is False:
                print("[WARNING] video requires enable_cameras; overriding to True.")
            args["enable_cameras"] = True
        return args

    def get_hydra_overrides(self, exclude_keys: set[str] | None = None) -> list[str]:
        """Generate Hydra CLI override arguments from this config.

        This is the KEY method for unified config handling. Both standalone
        training and Ray tuning use these overrides to configure env/agent.

        Note: Only generates overrides for fields that exist in the base config
        from the registry (Hydra struct mode). Additional fields are merged
        directly in the training script.

        Args:
            exclude_keys: Set of full dot-notation keys to exclude (e.g., {"agent.policy_kwargs.net_arch"})
                         Useful when Ray Tune provides these overrides to avoid duplicates.

        Returns:
            List of Hydra override strings like ["env.seed=42", "agent.learning_rate=0.0003"]
        """
        if exclude_keys is None:
            exclude_keys = set()

        overrides = []

        # Environment overrides
        # NOTE: We do NOT override env.seed via Hydra because the base config
        # has seed=None and Hydra strict mode rejects type changes. Seed is
        # set directly in the training script instead.

        if self.num_envs is not None:
            overrides.append(f"env.scene.num_envs={self.num_envs}")

        device = self.app_launcher.get("device")
        if device is not None:
            overrides.append(f"env.sim.device={device}")

        # Agent overrides - only fields that exist in base sb3_sac_cfg.yaml
        # to avoid Hydra struct mode errors. Other fields are merged directly.
        safe_agent_fields = {
            "n_timesteps",
            "learning_rate",
            "buffer_size",
            "learning_starts",
            "batch_size",
            "tau",
            "gamma",
            "train_freq",
            "gradient_steps",
            "ent_coef",
            "target_entropy",
            "target_update_interval",
            "normalize_input",
            "policy_kwargs",  # nested dict
        }

        agent_overrides = []
        for key, value in self.agent_cfg.items():
            if key in safe_agent_fields:
                agent_overrides.extend(
                    self._flatten_dict_to_overrides(
                        {key: value}, prefix="agent", exclude_keys=exclude_keys
                    )
                )

        overrides.extend(agent_overrides)

        return overrides

    def _flatten_dict_to_overrides(
        self, d: dict[str, Any], prefix: str = "", exclude_keys: set[str] | None = None
    ) -> list[str]:
        """Recursively flatten a dict into Hydra override format.

        Args:
            d: Dictionary to flatten
            prefix: Key prefix for nested dicts
            exclude_keys: Set of full dot-notation keys to exclude

        Example: {"policy_kwargs": {"net_arch": [64, 64]}} with prefix="agent"
        becomes: ["agent.policy_kwargs.net_arch=[64,64]"]
        """
        if exclude_keys is None:
            exclude_keys = set()

        overrides = []
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Skip excluded keys
            if full_key in exclude_keys:
                continue

            if isinstance(value, dict):
                # Recurse into nested dicts
                overrides.extend(
                    self._flatten_dict_to_overrides(value, full_key, exclude_keys)
                )
            elif isinstance(value, list):
                # Format lists for Hydra
                formatted = "[" + ",".join(str(v) for v in value) + "]"
                overrides.append(f"{full_key}={formatted}")
            elif isinstance(value, bool):
                # Hydra expects lowercase booleans
                overrides.append(f"{full_key}={str(value).lower()}")
            elif isinstance(value, str):
                # Quote strings that might have special characters
                if " " in value or "," in value:
                    overrides.append(f"'{full_key}={value}'")
                else:
                    overrides.append(f"{full_key}={value}")
            elif value is not None:
                overrides.append(f"{full_key}={value}")

        return overrides

    def get_agent_cfg(self) -> dict[str, Any]:
        """Get a copy of the agent configuration dict.

        This replaces the need for loading from sb3_sac_cfg_entry_point.
        """
        cfg = dict(self.agent_cfg)

        # Ensure n_timesteps is set
        if "n_timesteps" not in cfg:
            print("[ERROR] agent.n_timesteps must be specified in config.")
            sys.exit(1)

        return cfg

    # Keep these for backward compatibility but mark as deprecated
    def apply_to_env_cfg(self, env_cfg) -> None:
        """DEPRECATED: Use get_hydra_overrides() instead for unified config handling."""
        # Set seed
        if self.seed is not None:
            seed_value = self._coerce_int(self.seed, "seed")
            if seed_value == -1:
                seed_value = random.randint(0, 10000)
                self.seed = seed_value  # Update for logging
            env_cfg.seed = seed_value

        # Set num_envs
        if self.num_envs is not None:
            env_cfg.scene.num_envs = self.num_envs

        # Set device
        device = self.app_launcher.get("device")
        if device is not None:
            env_cfg.sim.device = device

    def apply_to_agent_cfg(self, agent_cfg: dict) -> None:
        """DEPRECATED: Use get_hydra_overrides() instead for unified config handling."""
        # Set seed
        if self.seed is not None:
            seed_value = self._coerce_int(self.seed, "seed")
            agent_cfg["seed"] = seed_value

        # Set device
        device = self.app_launcher.get("device")
        if device is not None:
            agent_cfg["device"] = device

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "task": self.task,
            "env": {
                "num_envs": self.num_envs,
                "seed": self.seed,
                "max_velocity": self.max_velocity,
            },
            "agent": self.agent_cfg,
            "her": self.her_cfg,
            "two_stage": self.two_stage_cfg,
            "training": {
                "log_interval": self.log_interval,
                "checkpoint": self.checkpoint,
                "export_io_descriptors": self.export_io_descriptors,
                "video": self.video_cfg,
            },
            "wandb": self.wandb_cfg,
            "app_launcher": self.app_launcher,
            "config_path": str(self.config_path) if self.config_path else None,
        }


# Backward compatibility alias
TrainConfig = ExperimentConfig

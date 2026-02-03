"""Training configuration helpers for SB3 scripts."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    task: str = "Klask-Rl-SAC-v0"
    agent: str = "sb3_sac_cfg_entry_point"
    num_envs: int | None = None
    seed: int | None = 42
    log_interval: int = 1000
    checkpoint: str | None = None
    max_timesteps: int | None = 10_000_000
    export_io_descriptors: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    keep_all_info: bool = False
    max_velocity: float = 0.2  # [m/s] Maximum velocity for player movement
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    # HER (Hindsight Experience Replay) settings
    use_her: bool = False  # Enable HER replay buffer
    her_goal_selection_strategy: str = (
        "future"  # 'future', 'final', 'episode', 'random'
    )
    her_n_sampled_goal: int = (
        4  # Number of virtual transitions to create per real transition
    )
    her_achieved_goal_indices: list[int] | None = (
        None  # [start, end] indices for achieved_goal in obs
    )
    her_desired_goal_indices: list[int] | None = (
        None  # [start, end] indices for desired_goal in obs
    )
    her_distance_threshold: float = 0.02  # Distance threshold for goal achievement
    her_reward_scale: float | None = (
        None  # Reward scale for HER (REQUIRED when use_her=True) - DEPRECATED: use her_env_reward_scale and her_wrapper_reward_scale
    )
    her_env_reward_scale: float | None = (
        None  # Environment reward weight for collision (when use_her=True)
    )
    her_wrapper_reward_scale: float | None = (
        None  # HER wrapper reward scale for compute_reward during hindsight relabeling (when use_her=True)
    )

    # Two-Stage HER settings (for goal-scoring task)
    # Note: rewards are handled by env's RewardsCfgTwoStageHer, not the wrapper
    use_two_stage_her: bool = False  # Enable two-stage HER wrapper
    two_stage_player_pos_indices: list[int] | None = (
        None  # [start, end] indices for player position in obs
    )
    two_stage_ball_pos_indices: list[int] | None = (
        None  # [start, end] indices for ball position in obs
    )
    two_stage_goal_pos_indices: list[int] | None = (
        None  # [start, end] indices for goal position in obs (from TwoStageHerObservationsCfg)
    )
    two_stage_ball_hit_threshold: float = 0.02  # Distance for ball hit detection
    two_stage_goal_score_threshold: float = 0.025  # Distance for goal scoring
    two_stage_ball_hit_reward: float | None = (
        None  # Reward for ball hit (REQUIRED when use_two_stage_her=True) - DEPRECATED: use two_stage_ball_hit_env_reward and two_stage_ball_hit_wrapper_reward
    )
    two_stage_goal_score_reward: float | None = (
        None  # Reward for goal scoring (REQUIRED when use_two_stage_her=True) - DEPRECATED: use two_stage_goal_score_env_reward and two_stage_goal_score_wrapper_reward
    )
    two_stage_ball_hit_env_reward: float | None = (
        None  # Environment reward weight for ball hit collision (when use_two_stage_her=True)
    )
    two_stage_ball_hit_wrapper_reward: float | None = (
        None  # HER wrapper reward for ball hit during hindsight relabeling (when use_two_stage_her=True)
    )
    two_stage_goal_score_env_reward: float | None = (
        None  # Environment reward weight for goal scoring (when use_two_stage_her=True)
    )
    two_stage_goal_score_wrapper_reward: float | None = (
        None  # HER wrapper reward for goal scoring during hindsight relabeling (when use_two_stage_her=True)
    )

    app_launcher: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _coerce_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            print(
                f"[ERROR] Training config '{field_name}' must be an integer, got bool."
            )
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
                        f"[ERROR] Training config '{field_name}' must be numeric, got: {value!r}"
                    )
                    sys.exit(1)
        print(
            f"[ERROR] Training config '{field_name}' must be numeric, got: {type(value).__name__}"
        )
        sys.exit(1)

    @staticmethod
    def _load_config_file(path: Path) -> dict[str, Any]:
        if not path.is_file():
            print(f"[ERROR] Training config not found: {path}")
            sys.exit(1)
        suffix = path.suffix.lower()
        try:
            with path.open("r", encoding="utf-8") as f:
                if suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(f)
                elif suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {suffix}")
        except Exception as exc:
            print(f"[ERROR] Failed to load training config: {path}\n{exc}")
            sys.exit(1)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            print(
                f"[ERROR] Training config must be a mapping, got: {type(data).__name__}"
            )
            sys.exit(1)
        return data

    @classmethod
    def from_file(cls, path: Path) -> "TrainConfig":
        data = cls._load_config_file(path)
        app_launcher = data.pop("app_launcher", {}) or {}
        if not isinstance(app_launcher, dict):
            print("[ERROR] Training config 'app_launcher' must be a mapping.")
            sys.exit(1)
        valid_fields = {f.name for f in fields(cls) if f.name != "app_launcher"}
        unknown = set(data) - valid_fields
        if unknown:
            print(f"[ERROR] Unknown training config fields: {sorted(unknown)}")
            sys.exit(1)
        return cls(app_launcher=app_launcher, **data)

    def setup_cuda_visibility(self) -> None:
        """Set CUDA_VISIBLE_DEVICES based on device in app_launcher config.
        Must be called BEFORE AppLauncher initialization to prevent Isaac Sim
        from allocating memory on all GPUs.
        """
        import os

        device = self.app_launcher.get("device")
        if device and isinstance(device, str) and device.startswith("cuda:"):
            try:
                gpu_id = device.split(":")[1]
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                print(
                    f"[INFO] Set CUDA_VISIBLE_DEVICES={gpu_id} (Physical GPU {gpu_id} will appear as cuda:0 to the application)"
                )
                # Update config to use cuda:0 since we've remapped the GPU
                self.app_launcher["device"] = "cuda:0"
            except (IndexError, ValueError):
                print(f"[WARNING] Could not parse GPU ID from device: {device}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def app_launcher_args(self) -> dict[str, Any]:
        args = dict(self.app_launcher)
        if self.video:
            if args.get("enable_cameras") is False:
                print(
                    "[WARNING] video=true requires enable_cameras; overriding to True."
                )
            args["enable_cameras"] = True
        return args

    def apply(self, env_cfg, agent_cfg) -> None:
        seed_value = agent_cfg.get("seed") if isinstance(agent_cfg, dict) else None
        if self.seed is not None:
            seed_value = self._coerce_int(self.seed, "seed")
            if seed_value == -1:
                seed_value = random.randint(0, 10000)
            agent_cfg["seed"] = seed_value
            self.seed = seed_value
        if seed_value is not None:
            env_cfg.seed = seed_value

        if self.num_envs is not None:
            env_cfg.scene.num_envs = self.num_envs

        if self.max_timesteps is not None:
            self.max_timesteps = self._coerce_int(self.max_timesteps, "max_timesteps")
            agent_cfg["n_timesteps"] = self.max_timesteps
        elif "n_timesteps" not in agent_cfg:
            print(
                "[ERROR] Training config missing max_timesteps and agent config missing n_timesteps."
            )
            sys.exit(1)

        device = self.app_launcher.get("device")
        if device is not None:
            env_cfg.sim.device = device
            agent_cfg["device"] = device

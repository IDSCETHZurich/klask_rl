"""Playback configuration helpers for SB3 scripts."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from train_config import TrainConfig


@dataclass
class PlayConfig:
    num_envs: int | None = None
    play_checkpoint: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000

    @classmethod
    def from_file(cls, path: Path) -> "PlayConfig":
        data = TrainConfig._load_config_file(path)
        valid_fields = {f.name for f in fields(cls)}
        unknown = set(data) - valid_fields
        if unknown:
            print(f"[ERROR] Unknown play config fields: {sorted(unknown)}")
            sys.exit(1)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply(self, env_cfg) -> None:
        if self.num_envs is not None:
            env_cfg.scene.num_envs = TrainConfig._coerce_int(self.num_envs, "num_envs")

"""Playback configuration helpers for SB3 scripts."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PlayConfig:
    num_envs: int | None = None
    play_checkpoint: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000

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
        return data if data else {}

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
                return int(float(value))
            except ValueError:
                print(f"[ERROR] Config '{field_name}' must be numeric, got: {value!r}")
                sys.exit(1)
        print(
            f"[ERROR] Config '{field_name}' must be numeric, got: {type(value).__name__}"
        )
        sys.exit(1)

    @classmethod
    def from_file(cls, path: Path) -> "PlayConfig":
        data = cls._load_yaml(path)
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
            env_cfg.scene.num_envs = self._coerce_int(self.num_envs, "num_envs")

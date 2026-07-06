"""Config loading. One experiment = one YAML in configs/. No magic numbers in scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Thin dict wrapper allowing attribute access and dotted .get with defaults."""

    __getattr__ = dict.get

    def require(self, key: str) -> Any:
        if key not in self:
            raise KeyError(f"config is missing required key: {key!r}")
        return self[key]


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a Config (dict subclass)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a YAML mapping, got {type(data).__name__}")
    cfg = Config(data)
    cfg["_config_name"] = path.stem
    cfg["_config_path"] = str(path)
    return cfg

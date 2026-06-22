from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else repo_root() / "configs" / "default.json"
    with cfg_path.open() as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(base) if base else repo_root()
    return (root / p).resolve()

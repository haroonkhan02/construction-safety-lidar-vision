"""YAML config loading with dotted-key overrides (`--set lidar.dbscan_eps=0.6`)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(os.environ.get("SITESAFETY_CONFIG", "configs/default.yaml"))


def load_config(path: Path | str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def pick_device(preferred: str = "auto") -> str:
    """'mps' on Apple silicon, 'cuda' if present, else 'cpu'."""
    if preferred != "auto":
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

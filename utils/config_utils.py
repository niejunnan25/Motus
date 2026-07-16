"""Small, explicit YAML composition helpers used by experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def load_config_with_base(
    config_path: str | Path,
    *,
    _stack: Iterable[Path] = (),
) -> DictConfig:
    """Load an OmegaConf YAML and recursively merge its optional `_base_` file."""
    path = Path(config_path).expanduser().resolve()
    stack = tuple(_stack)
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Recursive config inheritance detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    current = OmegaConf.load(path)
    base_value = current.pop("_base_", None)
    if base_value is None:
        return current
    base_path = Path(str(base_value))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = load_config_with_base(base_path, _stack=(*stack, path))
    return OmegaConf.merge(base, current)

"""Algorithm module for pluggable RL algorithms.

New algorithms register themselves via ``@register_algo("name")`` on their
config dataclass (or any sentinel class).  The adapter reads ``algo_names()``
to build the validation whitelist, so adding a new algorithm only requires:

  1. Registering it here with ``@register_algo``.
  2. Adding its name to ``config/registry.yaml``.
  3. Writing the trainer implementation.
"""

from __future__ import annotations

from typing import Any

_ALGO_REGISTRY: dict[str, type] = {}


def register_algo(name: str):
    """Class decorator that registers an algo config class under *name*."""
    def decorator(cls: type) -> type:
        _ALGO_REGISTRY[name] = cls
        return cls
    return decorator


def algo_names() -> tuple[str, ...]:
    """Return the names of all registered algorithms."""
    return tuple(_ALGO_REGISTRY)


def get_algo_config_cls(name: str) -> type:
    """Return the config class registered under *name*, or raise ValueError."""
    if name not in _ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algo {name!r}; registered: {algo_names()}"
        )
    return _ALGO_REGISTRY[name]


__all__ = [
    "register_algo",
    "algo_names",
    "get_algo_config_cls",
]

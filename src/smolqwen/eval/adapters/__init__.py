"""Discover benchmark adapters without a central name-to-class switch."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters.base import BenchmarkAdapter

AdapterFactory = Callable[[EvalConfig], BenchmarkAdapter]


@lru_cache(maxsize=1)
def adapter_factories() -> dict[str, AdapterFactory]:
    """Load adapter modules that publish ``ADAPTER_NAME`` and ``create_adapter``."""

    factories: dict[str, AdapterFactory] = {}
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name in {"base"}:
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        name = getattr(module, "ADAPTER_NAME", None)
        factory: Any = getattr(module, "create_adapter", None)
        if name is None and factory is None:
            continue
        if not isinstance(name, str) or not callable(factory):
            raise RuntimeError(
                f"adapter module {module.__name__} must define a string ADAPTER_NAME "
                "and callable create_adapter"
            )
        if name in factories:
            raise RuntimeError(f"duplicate evaluation adapter name: {name}")
        factories[name] = factory
    return factories


def create_adapter(name: str, config: EvalConfig) -> BenchmarkAdapter:
    """Construct one discovered adapter by its stable config/CLI name."""

    try:
        factory = adapter_factories()[name]
    except KeyError as exc:
        available = ", ".join(sorted(adapter_factories()))
        raise ValueError(
            f"unsupported evaluation adapter {name!r}; available: {available}"
        ) from exc
    return factory(config)

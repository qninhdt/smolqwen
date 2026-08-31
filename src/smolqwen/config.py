"""Config resolution: a pure function from (stage, profile, overrides) to a model.

``resolve(stage, profile, overrides) -> StrictModel``. Pure means testable without
a GPU, without a network, and without a token -- which is the whole point of
`--dry-run` catching a typo at load rather than thirty minutes into a run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from smolqwen.config_models import (
    BUDGET_SEEDED_FIELDS,
    PROFILES,
    STAGE_MODELS,
    STAGES,
    ConfigError,
    ProfileConfig,
    StrictModel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_BUDGETS_PATH = REPO_ROOT / "artifacts" / "data" / "budgets.json"


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` onto `base`; the overlay's leaves win.

    Mappings merge, everything else replaces. A list is a leaf: a profile that
    sets `compile_exclude_patterns` replaces the base list rather than appending
    to it, because a half-overridden exclusion list is worse than either.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config file must contain a mapping at the top level: {path}")
    return loaded


def _resolve_annotation(model: type[StrictModel], path: Sequence[str]) -> Any:
    """Walk a dotted override path through nested models to the target annotation.

    Raising on an unknown segment is the point: `--override sft.trainig.lr=1e-4`
    must fail at load with the model and key named, not be silently accepted and
    then dropped by `extra="forbid"` with a less useful message.
    """
    current: Any = model
    for index, segment in enumerate(path):
        fields: Mapping[str, FieldInfo] | None = getattr(current, "model_fields", None)
        if fields is None:
            prefix = ".".join(path[:index])
            raise ConfigError(
                f"override path '{'.'.join(path)}' goes too deep: '{prefix}' is not a section"
            )
        if segment not in fields:
            available = ", ".join(sorted(fields))
            prefix = ".".join(path[:index]) or "<root>"
            raise ConfigError(
                f"unknown config key '{segment}' under '{prefix}'. available keys: {available}"
            )
        current = fields[segment].annotation
    return current


def _coerce_scalar(raw: str, annotation: Any) -> Any:
    """Coerce a CLI string to the target field's annotated type.

    Without this, `--override sft.training.learning_rate=1e-4` lands as the string
    ``"1e-4"``. Pydantic would coerce that one, but `bool` is the trap: ``bool("false")``
    is ``True``, so a `--override ...bf16=false` would silently enable the flag.
    """
    origin = get_origin(annotation)
    if origin is not None:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        # Optional[X] / X | None: an explicit "null" clears the field.
        if raw.lower() in {"null", "none"} and len(args) < len(get_args(annotation)):
            return None
        if origin in (list, tuple, Sequence):
            return [item.strip() for item in raw.split(",") if item.strip()]
        if len(args) == 1:
            return _coerce_scalar(raw, args[0])
        # A union of scalars (e.g. `str | Sequence[str]`): hand the raw string to
        # pydantic and let the model's own validation decide.
        return raw

    if annotation is bool:
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ConfigError(f"cannot parse '{raw}' as a boolean")
    if annotation is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"cannot parse '{raw}' as an integer") from exc
    if annotation is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"cannot parse '{raw}' as a float") from exc
    return raw


def parse_override(model: type[StrictModel], spec: str) -> dict[str, Any]:
    """Turn ``SECTION.KEY=VALUE`` into a nested dict with the value already coerced."""
    if "=" not in spec:
        raise ConfigError(f"override must be SECTION.KEY=VALUE, got: {spec!r}")
    dotted, raw = spec.split("=", 1)
    path = [segment for segment in dotted.strip().split(".") if segment]
    if not path:
        raise ConfigError(f"override is missing a key: {spec!r}")

    annotation = _resolve_annotation(model, path)
    if getattr(annotation, "model_fields", None) is not None:
        raise ConfigError(f"override path '{dotted}' names a section, not a value; append a key")
    value = _coerce_scalar(raw, annotation)

    nested: dict[str, Any] = {}
    cursor = nested
    for segment in path[:-1]:
        child: dict[str, Any] = {}
        cursor[segment] = child
        cursor = child
    cursor[path[-1]] = value
    return nested


def load_budgets(path: Path | None = None) -> dict[str, Any]:
    """Read the Phase 2 budget artifact into a profile overlay.

    Absent is fine before Phase 2 has run -- the profile defaults apply and the
    caps are seeded later. Present and malformed is not fine: a stage that thinks
    it is honouring a measured cap while reading a default is exactly the silent
    failure the file exists to prevent.
    """
    budgets_path = path or DEFAULT_BUDGETS_PATH
    if not budgets_path.is_file():
        return {}
    with budgets_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ConfigError(f"budgets file must contain a mapping: {budgets_path}")

    recommended = payload.get("recommended", payload)
    if not isinstance(recommended, Mapping):
        raise ConfigError(f"budgets 'recommended' must be a mapping: {budgets_path}")

    overlay: dict[str, Any] = {}
    for budget_key, profile_field in BUDGET_SEEDED_FIELDS.items():
        if budget_key in recommended:
            overlay[profile_field] = recommended[budget_key]
    return {"profile": overlay} if overlay else {}


def _profile_cap_violations(profile: ProfileConfig, budgets: Mapping[str, Any]) -> list[str]:
    """Report profile fields that exceed the measured budget cap.

    The budget bounds the distribution; a sizing sweep may only find what fits
    *under* it. Raising a cap because a bigger batch happened to fit would mean
    training on a slice the profiler never characterised.
    """
    overlay = budgets.get("profile", {})
    if not isinstance(overlay, Mapping):
        return []
    violations: list[str] = []
    for field, cap in overlay.items():
        actual = getattr(profile, field, None)
        if isinstance(actual, int) and isinstance(cap, int) and actual > cap:
            violations.append(f"profile.{field}={actual} exceeds budgets.json cap {cap}")
    return violations


def resolve(
    stage: str,
    profile: str | None = None,
    overrides: Sequence[str] = (),
    config_path: Path | None = None,
    config_dir: Path | None = None,
    budgets_path: Path | None = None,
) -> StrictModel:
    """Resolve one stage's config.

    Merge order is ``budgets.json -> base -> profile -> overrides``, deepest wins.
    """
    if stage not in STAGES:
        raise ConfigError(f"unknown stage '{stage}'; expected one of {', '.join(STAGES)}")
    model = STAGE_MODELS[stage]

    directory = config_dir or DEFAULT_CONFIG_DIR
    base_path = config_path or (directory / "base" / f"{stage}.yaml")
    merged: dict[str, Any] = deep_merge(load_budgets(budgets_path), _load_yaml(base_path))

    if profile is not None:
        if profile not in PROFILES:
            raise ConfigError(f"unknown profile '{profile}'; expected one of {', '.join(PROFILES)}")
        profile_payload = _load_yaml(directory / "profiles" / f"{profile}.yaml")
        # A profile YAML holds sizing fields at its top level; nest them under the
        # stage model's `profile` section so a profile cannot reach a semantic key.
        merged = deep_merge(merged, {"profile": profile_payload})
        if stage == "serve":
            # Serving is the intentional exception to sizing-only GPU profiles:
            # sm89 L4 and sm80 A100 require different quantization strategies.
            # Keep those measured overlays separate from training-owned fields.
            serving_profile = directory / "serving" / f"{profile}.yaml"
            if serving_profile.is_file():
                merged = deep_merge(merged, _load_yaml(serving_profile))

    for spec in overrides:
        merged = deep_merge(merged, parse_override(model, spec))

    try:
        resolved = model.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid {stage} config: {exc}") from exc

    budgets = load_budgets(budgets_path)
    violations = _profile_cap_violations(getattr(resolved, "profile", ProfileConfig()), budgets)
    if violations:
        raise ConfigError("; ".join(violations))
    return resolved


def resolved_summary(config: StrictModel) -> str:
    """A stable, readable dump for `--dry-run` and for the W&B run config."""
    return json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True)

"""API configuration loaded from ``configs/api.yaml``.

The configuration is read once at module import. Tests can override
the resulting values via :func:`apply_overrides` to keep the dev test
suite hermetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "api.yaml"


@dataclass(frozen=True)
class ApiConfig:
    """Settings consumed by the dev API."""

    supported_languages: set[str] = field(default_factory=lambda: {"en"})
    min_text_chars_for_language_check: int = 20
    min_language_score: float = 0.0

    @property
    def supported_languages_list(self) -> list[str]:
        return sorted(self.supported_languages)


def _coerce_supported_languages(raw: object) -> set[str]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("api.supported_languages must be a list of ISO 639-1 codes")
    codes: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str) or len(entry) != 2:
            raise ValueError(f"api.supported_languages must contain 2-letter codes, got {entry!r}")
        codes.add(entry.lower())
    if not codes:
        raise ValueError("api.supported_languages must contain at least one code")
    return codes


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"api config at {path} must be a mapping")
    return data


def load_api_config(path: Path | None = None) -> ApiConfig:
    """Load and validate the API configuration from a YAML file."""

    data = _load_yaml(path or DEFAULT_CONFIG_PATH)
    api_section = data.get("api") if isinstance(data, dict) else None
    if not isinstance(api_section, dict):
        api_section = {}

    return ApiConfig(
        supported_languages=_coerce_supported_languages(
            api_section.get("supported_languages", ["en"])
        ),
        min_text_chars_for_language_check=int(
            api_section.get("min_text_chars_for_language_check", 20)
        ),
        min_language_score=float(api_section.get("min_language_score", 0.0)),
    )


@lru_cache(maxsize=1)
def _cached_config() -> ApiConfig:
    override = os.environ.get("TRIAGE_ML_API_CONFIG")
    path = Path(override) if override else None
    return load_api_config(path)


def get_api_config() -> ApiConfig:
    """Return the process-wide API configuration (cached)."""

    return _cached_config()


def reset_api_config_cache() -> None:
    """Clear the LRU cache (used by tests when changing ``TRIAGE_ML_API_CONFIG``)."""

    _cached_config.cache_clear()

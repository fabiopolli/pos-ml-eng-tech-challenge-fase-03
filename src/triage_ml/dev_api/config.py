"""API configuration loaded from ``configs/api.yaml``.

The configuration is loaded lazily and cached process-wide. Tests clear
the cache when changing the override path.
"""

from __future__ import annotations

import math
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

    supported_languages: frozenset[str] = field(default_factory=lambda: frozenset({"en"}))
    min_text_chars_for_language_check: int = 20
    min_language_score: float = 0.0

    def __post_init__(self) -> None:
        languages = _coerce_supported_languages(self.supported_languages)
        object.__setattr__(self, "supported_languages", frozenset(languages))
        if (
            isinstance(self.min_text_chars_for_language_check, bool)
            or not isinstance(self.min_text_chars_for_language_check, int)
            or not 1 <= self.min_text_chars_for_language_check <= 20_000
        ):
            raise ValueError("api.min_text_chars_for_language_check must be between 1 and 20000")
        score = self.min_language_score
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("api.min_language_score must be a finite number between 0 and 1")

    @property
    def supported_languages_list(self) -> list[str]:
        return sorted(self.supported_languages)


def _coerce_supported_languages(raw: object) -> set[str]:
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError("api.supported_languages must be a list of ISO 639-1 codes")
    codes: set[str] = set()
    for entry in raw:
        if (
            not isinstance(entry, str)
            or len(entry) != 2
            or not entry.isascii()
            or not entry.isalpha()
        ):
            raise ValueError(f"api.supported_languages must contain 2-letter codes, got {entry!r}")
        codes.add(entry.lower())
    if not codes:
        raise ValueError("api.supported_languages must contain at least one code")
    return codes


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"api config does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"api config at {path} must be a mapping")
    return data


def load_api_config(path: Path | None = None) -> ApiConfig:
    """Load and validate the API configuration from a YAML file."""

    data = _load_yaml(path or DEFAULT_CONFIG_PATH)
    api_section = data.get("api")
    if api_section is None:
        api_section = {}
    elif not isinstance(api_section, dict):
        raise ValueError("api config section 'api' must be a mapping")

    min_chars = api_section.get("min_text_chars_for_language_check", 20)
    min_score = api_section.get("min_language_score", 0.0)
    if isinstance(min_chars, bool) or not isinstance(min_chars, int):
        raise ValueError("api.min_text_chars_for_language_check must be an integer")
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise ValueError("api.min_language_score must be numeric")

    return ApiConfig(
        supported_languages=_coerce_supported_languages(
            api_section.get("supported_languages", ["en"])
        ),
        min_text_chars_for_language_check=min_chars,
        min_language_score=float(min_score),
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

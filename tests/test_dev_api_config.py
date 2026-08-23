"""Validation tests for the dev API configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from triage_ml.dev_api.config import ApiConfig, load_api_config


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_api_config_rejects_invalid_language_score(score: float) -> None:
    with pytest.raises(ValueError, match="min_language_score"):
        ApiConfig(min_language_score=score)


def test_api_config_rejects_invalid_minimum_length() -> None:
    with pytest.raises(ValueError, match="min_text_chars"):
        ApiConfig(min_text_chars_for_language_check=0)


def test_api_config_rejects_non_mapping_api_section(tmp_path: Path) -> None:
    path = tmp_path / "api.yaml"
    path.write_text("api: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_api_config(path)


def test_api_config_override_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_api_config(tmp_path / "missing.yaml")

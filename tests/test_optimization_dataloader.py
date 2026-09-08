"""Tests for ``triage_ml.optimization.dataloader`` — Fase 2 / Etapa 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triage_ml.optimization import iter_dataset_slices
from triage_ml.optimization.dataloader import (
    DatasetSlice,
    load_canonical_for_slice,
    slice_name_for,
)


def _write_minimal_csv(path: Path, *, rows_per_label: int = 1500) -> Path:
    rows = [
        f"distinct medical abstract number {i:04d} about cardiology"
        for i in range(rows_per_label * 2)
    ]
    df = pd.DataFrame(
        {
            "medical_abstract": rows,
            "condition_label": [1] * rows_per_label + [2] * rows_per_label,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _config_with(sample_size: int, sizing: list[int] | None = None) -> dict:
    config: dict = {"sample_size": sample_size, "random_state": 42}
    if sizing is not None:
        config["dataset_sizing"] = sizing
    return config


def test_iter_dataset_slices_returns_default_when_unspecified(tmp_path: Path) -> None:
    config = _config_with(5_000)
    _write_minimal_csv(tmp_path / "unused.csv")

    slices = list(iter_dataset_slices(config, base_csv=tmp_path / "unused.csv"))

    assert slices == [DatasetSlice(name="slice_5k", sample_size=5_000)]


def test_iter_dataset_slices_normalises_catalog(tmp_path: Path) -> None:
    config = _config_with(5_000, sizing=[14000, 5000, 10000])
    _write_minimal_csv(tmp_path / "unused.csv")

    slices = list(iter_dataset_slices(config, base_csv=tmp_path / "unused.csv"))

    assert [slice.sample_size for slice in slices] == [5_000, 10_000, 14_000]
    assert [slice.name for slice in slices] == ["slice_5k", "slice_10k", "slice_14k"]


def test_iter_dataset_slices_rejects_missing_base_csv(tmp_path: Path) -> None:
    config = _config_with(5_000)
    with pytest.raises(FileNotFoundError):
        list(iter_dataset_slices(config, base_csv=tmp_path / "missing.csv"))


@pytest.mark.parametrize("sizing", [[0], [-1], [5_000, "ten"]])
def test_iter_dataset_slices_rejects_invalid_catalog(sizing: list, tmp_path: Path) -> None:
    config = _config_with(5_000, sizing=sizing)
    _write_minimal_csv(tmp_path / "unused.csv")
    with pytest.raises(ValueError):
        list(iter_dataset_slices(config, base_csv=tmp_path / "unused.csv"))


def test_iter_dataset_slices_honours_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGE_DATASET_SLICES", "8000,5000")
    config = _config_with(5_000, sizing=[14_000, 5_000])
    _write_minimal_csv(tmp_path / "unused.csv")

    slices = list(iter_dataset_slices(config, base_csv=tmp_path / "unused.csv"))

    assert [slice.sample_size for slice in slices] == [5_000, 8_000]


def test_iter_dataset_slices_rejects_env_override_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGE_DATASET_SLICES", "   ,  ,")
    config = _config_with(5_000)
    _write_minimal_csv(tmp_path / "unused.csv")
    with pytest.raises(ValueError):
        list(iter_dataset_slices(config, base_csv=tmp_path / "unused.csv"))


def test_slice_name_for_picks_human_readable_label() -> None:
    assert slice_name_for(5_000) == "slice_5k"
    assert slice_name_for(14_000) == "slice_14k"
    assert slice_name_for(800) == "slice_800"


def test_load_canonical_for_slice_uses_canonical_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _write_minimal_csv(tmp_path / "data" / "train.csv")

    canonical, report = load_canonical_for_slice(
        {"random_state": 7}, base_csv=base, sample_size=2_000, random_state=7
    )

    assert report.eligible_rows == 3_000
    assert len(canonical) == 2_000
    assert "text" in canonical.columns and "target" in canonical.columns


def test_dataset_slice_post_init_validations() -> None:
    with pytest.raises(ValueError):
        DatasetSlice(name="slice_5k", sample_size=0)
    with pytest.raises(ValueError):
        DatasetSlice(name="has space", sample_size=5_000)


def test_iter_dataset_slices_does_not_load_csv(tmp_path: Path) -> None:
    """When ``base_csv`` is not supplied the iterator must not touch the FS."""

    config = _config_with(5_000)
    slices = list(iter_dataset_slices(config))
    assert len(slices) == 1 and slices[0].sample_size == 5_000

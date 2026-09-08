"""Tests for ``triage_ml.optimization.registry`` — Fase 2 / Etapa 5."""

from __future__ import annotations

from pathlib import Path

import pytest

from triage_ml.optimization import (
    AVAILABLE_VARIANTS,
    resolve_variant_loader,
    validate_variant_metadata,
)
from triage_ml.optimization.registry import (
    read_runtime_variant,
    resolve_artifact_for_variant,
)


def test_available_variants_is_scikit_and_onnx() -> None:
    assert AVAILABLE_VARIANTS == ("sklearn", "onnx")


def test_resolve_variant_loader_returns_callable() -> None:
    sklearn_loader = resolve_variant_loader("sklearn")
    onnx_loader = resolve_variant_loader("onnx")
    assert callable(sklearn_loader)
    assert callable(onnx_loader)


def test_resolve_variant_loader_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRIAGE_ML_MODEL_VARIANT", raising=False)
    with pytest.raises(ValueError):
        resolve_variant_loader("invalid")


def test_validate_variant_metadata_accepts_advertised_variant() -> None:
    metadata = {"available_variants": ["sklearn", "onnx"]}
    validate_variant_metadata(metadata, variant="onnx")


def test_validate_variant_metadata_rejects_unavailable_variant() -> None:
    metadata = {"available_variants": ["sklearn"]}
    with pytest.raises(ValueError):
        validate_variant_metadata(metadata, variant="onnx")


def test_validate_variant_metadata_rejects_missing_field() -> None:
    with pytest.raises(ValueError):
        validate_variant_metadata({}, variant="sklearn")


def test_read_runtime_variant_defaults_to_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRIAGE_ML_MODEL_VARIANT", raising=False)
    assert read_runtime_variant() == "sklearn"


@pytest.mark.parametrize("variant", ["sklearn", "onnx", "SKLEARN", "Onnx"])
def test_read_runtime_variant_accepts_declared_values(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGE_ML_MODEL_VARIANT", variant)
    assert read_runtime_variant() in AVAILABLE_VARIANTS


def test_read_runtime_variant_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_ML_MODEL_VARIANT", "pytorch")
    with pytest.raises(ValueError):
        read_runtime_variant()


def test_resolve_artifact_for_variant_returns_expected_filename() -> None:
    base = Path("models/20260905T171611Z-f2cb6f23f9cd")
    assert resolve_artifact_for_variant(base, "sklearn") == base / "model.joblib"
    assert (
        resolve_artifact_for_variant(base, "onnx", onnx_filename="model.onnx")
        == base / "model.onnx"
    )


def test_resolve_artifact_for_variant_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_artifact_for_variant(Path("models/x"), "random")

"""Tests for ``triage_ml.optimization.benchmark`` — Fase 2 / Etapa 5.

These tests cover the deterministic surfaces of the controlled benchmark and
avoid requiring the optional ``[optimization]`` dependency group.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triage_ml.optimization.benchmark import (
    BenchmarkResult,
    EnvironmentFingerprint,
    _agreement_rate,
    _macro_f1,
    _percentile,
    benchmark_predictor,
    capture_environment,
    write_benchmark_json,
)


class _FakePredictor:
    """Deterministic stand-in for an sklearn / ONNX predictor."""

    def __init__(self, prediction: int, latency_ms: float = 0.0) -> None:
        self._prediction = prediction
        self._latency_ms = latency_ms

    def predict(self, texts):  # pragma: no cover - smoke tested below
        import time

        if self._latency_ms:
            time.sleep(self._latency_ms / 1000.0)
        return [self._prediction] * len(texts)


def test_percentile_returns_a_value_between_min_and_max() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(samples, 50) == pytest.approx(3.0)
    assert _percentile(samples, 95) == pytest.approx(4.8, rel=0.01)


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError):
        _percentile([], 50)


def test_agreement_rate_counts_position_wise_matches() -> None:
    assert _agreement_rate([1, 2, 3, 4], [1, 2, 9, 4]) == 0.75


def test_macro_f1_handles_partial_overlap() -> None:
    score = _macro_f1([1, 2, 3], [1, 2, 4])
    assert 0.0 < score <= 1.0


def test_benchmark_predictor_returns_full_payload() -> None:
    predictor = _FakePredictor(prediction=1)
    result = benchmark_predictor(
        predictor,
        texts=["alpha"],
        variant="sklearn",
        reference_predictions=[1],
        reference_labels=[1],
        repetitions=4,
        warmup=1,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.variant == "sklearn"
    assert result.n_samples == 1
    assert result.repetitions == 4
    assert result.macro_f1 is not None and 0.0 <= result.macro_f1 <= 1.0
    assert result.class_agreement == 1.0  # matched reference exactly


def test_benchmark_predictor_validates_arguments() -> None:
    with pytest.raises(ValueError):
        benchmark_predictor(_FakePredictor(1), texts=["x"], repetitions=0, warmup=0, variant="x")
    with pytest.raises(ValueError):
        benchmark_predictor(_FakePredictor(1), texts=[], variant="x")
    with pytest.raises(ValueError):
        benchmark_predictor(_FakePredictor(1), texts=["x"], batch_size=0, variant="x")


def test_benchmark_quality_uses_the_complete_evaluation_set() -> None:
    result = benchmark_predictor(
        _FakePredictor(1),
        texts=["a", "b", "c"],
        variant="onnx",
        reference_predictions=[1, 1, 1],
        reference_labels=[1, 2, 1],
        repetitions=2,
        warmup=0,
    )

    assert result.n_samples == 3
    assert result.class_agreement == 1.0
    assert result.macro_f1 is not None and result.macro_f1 < 1.0


def test_capture_environment_handles_missing_optimization_groups() -> None:
    """``capture_environment`` must succeed even without onnx/sk2onnx/onnxruntime."""

    fingerprint = capture_environment(onnx_runtime_version=None)
    assert isinstance(fingerprint, EnvironmentFingerprint)
    assert fingerprint.python
    assert fingerprint.numpy


def test_write_benchmark_json_emits_payload(tmp_path: Path) -> None:
    sklearn_result = BenchmarkResult(
        variant="sklearn",
        n_samples=4,
        batch_size=1,
        repetitions=4,
        warmup=1,
        load_seconds=0.0,
        latency_mean_ms=1.0,
        latency_p50_ms=1.0,
        latency_p95_ms=1.0,
        latency_p99_ms=1.0,
        macro_f1=0.7,
        class_agreement=0.95,
        throughput_predictions_per_sec=1000.0,
    )
    environment = capture_environment(onnx_runtime_version=None)
    path = tmp_path / "reports" / "benchmarks" / "benchmark.json"

    written = write_benchmark_json(
        path,
        sklearn_result=sklearn_result,
        onnx_result=None,
        environment=environment,
        extra_metadata={"artifact_version": "test"},
    )

    assert written == path
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sklearn"]["variant"] == "sklearn"
    assert payload["onnx"] is None
    assert payload["extra_metadata"]["artifact_version"] == "test"

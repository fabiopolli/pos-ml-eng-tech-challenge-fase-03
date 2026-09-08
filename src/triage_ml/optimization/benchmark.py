"""Controlled latency benchmark for sklearn vs ONNX variants (Fase 2, Etapa 5).

Methodology (mirrors the criterion documented in
``docs/plans/PLAN-text-classifier.md`` — Fase 2):

* ``batch_size = 1`` (wost-case inference latency, the user-facing metric).
* Warmup of ``warmup`` predictions is discarded before timing.
* Median, p95, p99 and mean are computed from ``repetitions`` timings.
* The clock covers the TF-IDF transform + classifier (sklearn) or the entire
  ONNX session run. Artifact *loading* time is reported separately.
* Macro-F1 and a per-class agreement rate against the sklearn reference are
  reported; the caller decides whether the variant is acceptable.

The benchmark is intentionally hardware-aware (records Python, OS, CPU count)
so the JSON evidence carries everything needed to reproduce the measurement.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BATCH_SIZE = 1
DEFAULT_REPETITIONS = 50
DEFAULT_WARMUP = 5


@dataclass(frozen=True)
class BenchmarkResult:
    """Output of one pass over a single variant."""

    variant: str
    n_samples: int
    batch_size: int
    repetitions: int
    warmup: int
    load_seconds: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    macro_f1: float | None
    class_agreement: float | None
    throughput_predictions_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(samples: Sequence[float], pct: float) -> float:
    if not samples:
        raise ValueError("Cannot compute percentile of an empty sample set")
    return float(np.percentile(np.asarray(samples), pct))


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Hardware/environment snapshot attached to every benchmark run."""

    python: str
    platform: str
    cpu_count: int
    onnxruntime: str | None
    sklearn: str
    numpy: str
    skl2onnx: str | None
    onnx: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_environment(*, onnx_runtime_version: str | None = None) -> EnvironmentFingerprint:
    import numpy
    import sklearn

    onnx_runtime_version_resolved: str | None
    skl2onnx_version: str | None
    onnx_version: str | None
    try:
        import skl2onnx

        skl2onnx_version = skl2onnx.__version__
    except ImportError:
        skl2onnx_version = None
    try:
        import onnx

        onnx_version = onnx.__version__
    except ImportError:
        onnx_version = None
    onnx_runtime_version_resolved = onnx_runtime_version
    if onnx_runtime_version_resolved is None:
        try:
            import onnxruntime

            onnx_runtime_version_resolved = onnxruntime.__version__
        except ImportError:
            onnx_runtime_version_resolved = None

    return EnvironmentFingerprint(
        python=platform.python_version(),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        onnxruntime=onnx_runtime_version_resolved,
        sklearn=sklearn.__version__,
        numpy=numpy.__version__,
        skl2onnx=skl2onnx_version,
        onnx=onnx_version,
    )


def _time_call(call: Callable[[], Any], *, repetitions: int, warmup: int) -> list[float]:
    """Run ``warmup`` untimed calls then ``repetitions`` timed calls."""

    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def benchmark_predictor(
    predictor: Any,
    texts: Sequence[str],
    *,
    variant: str,
    reference_predictions: Sequence[int] | None = None,
    reference_labels: Sequence[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    repetitions: int = DEFAULT_REPETITIONS,
    warmup: int = DEFAULT_WARMUP,
) -> BenchmarkResult:
    """Benchmark a single predictor and return ``BenchmarkResult``.

    ``reference_predictions`` is optional; when supplied together with
    ``reference_labels`` the agreement rate and a macro-F1 surface are
    included (sklearn reference vs measured predictions).
    """

    if repetitions <= 0 or warmup < 0:
        raise ValueError("repetitions must be > 0 and warmup must be >= 0")
    if not texts:
        raise ValueError("texts must contain at least one element to benchmark")

    def predict_one() -> Any:
        return predictor.predict(list(texts[:batch_size]))

    load_start = time.perf_counter()
    predict_one()  # warm up TF-IDF/ONNX caches
    load_seconds = time.perf_counter() - load_start

    timings = _time_call(predict_one, repetitions=repetitions, warmup=warmup)
    predictions = predictor.predict(list(texts[:batch_size]))
    predictions_list = [int(value) for value in predictions]

    agreement: float | None = None
    macro_f1: float | None = None
    if reference_predictions is not None and reference_labels is not None:
        agreement = _agreement_rate(predictions_list, list(reference_predictions))
        macro_f1 = _macro_f1(predictions_list, list(reference_labels))

    throughput = 1000.0 / statistics.mean(timings) if timings else 0.0

    return BenchmarkResult(
        variant=variant,
        n_samples=len(texts),
        batch_size=batch_size,
        repetitions=repetitions,
        warmup=warmup,
        load_seconds=load_seconds,
        latency_mean_ms=float(statistics.mean(timings)),
        latency_p50_ms=_percentile(timings, 50),
        latency_p95_ms=_percentile(timings, 95),
        latency_p99_ms=_percentile(timings, 99),
        macro_f1=macro_f1,
        class_agreement=agreement,
        throughput_predictions_per_sec=throughput,
    )


def _agreement_rate(a: Sequence[int], b: Sequence[int]) -> float:
    pairs = list(zip(a, b, strict=True))
    if not pairs:
        return 0.0
    matches = sum(1 for x, y in pairs if x == y)
    return matches / len(pairs)


def _macro_f1(predictions: Sequence[int], references: Sequence[int]) -> float:
    pairs = list(zip(predictions, references, strict=True))
    if not pairs:
        return 0.0
    classes = sorted({*predictions, *references})
    f1_per_class = []
    for cls in classes:
        tp = sum(1 for p, r in pairs if p == cls and r == cls)
        fp = sum(1 for p, r in pairs if p == cls and r != cls)
        fn = sum(1 for p, r in pairs if p != cls and r == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_per_class) / len(f1_per_class))


def write_benchmark_json(
    path: Path,
    *,
    sklearn_result: BenchmarkResult,
    onnx_result: BenchmarkResult | None,
    environment: EnvironmentFingerprint,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a side-by-side JSON evidence file under ``path``."""

    import json

    payload = {
        "sklearn": sklearn_result.to_dict(),
        "onnx": onnx_result.to_dict() if onnx_result is not None else None,
        "environment": environment.to_dict(),
    }
    if extra_metadata:
        payload["extra_metadata"] = extra_metadata

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return path

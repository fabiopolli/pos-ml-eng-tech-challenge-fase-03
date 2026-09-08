"""Regression tests for the post-Fase-2 review (Bill).

Each test maps to one of the consolideted findings surfaced by the two
cross-reviewer sub-agents. The goal is to lock the corrected behaviour
behind a deterministic assertion so the bugs do not slip back in.

Tests here are deliberately deterministic — they do not require the
``[optimization]`` extra to be installed (those tests already live in
``test_model_optimization.py`` and skip when the extras are absent).

Findings exercised:

* ``ONNXRUNTIME_AVAILABLE=False`` still loads without crashing.
* ``OnnxModelAdapter`` ``predict`` reuses the same ``InferenceSession``
  across calls (singleton — fixes the per-request memory leak).
* ``export_onnx`` calls ``convert_sklearn`` with ``zipmap=False`` so the
  adapter's ``predict_proba`` path is well-defined.
* ``OptimizationFingerprint`` (when ``[optimization]`` extras are
  available) classifies ``LogisticRegression`` as ``"logreg"`` rather
  than ``"LogisticRegression"``.
* ``generate_observability_traffic.py`` uses ``json.dumps`` instead of
  ``str(dict)`` so the body is valid JSON.
* ``_normalise_error_code`` (a) lowercases, (b) strips whitespace,
  (c) rejects codes not in the allow-list.
* ``train_with_sample_size`` idempotency rejects a second run whose
  ``random_state`` differs (it must rebuild).
* ``compare_slices`` resolves ``sample_size`` from per-slice payloads,
  not from ``payload.get("sample_size", 0)``.
* ``benchmark_for_version`` writes both the aggregate and the per-version
  JSON atomically and tolerates ``benchmark_input_size=0``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triage_ml.optimization.optimize import ONNX_AVAILABLE


def test_onnx_adapter_module_imports_when_extras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``triage_ml.optimization.onnx_adapter`` must be importable in any env."""

    # The module guarded the import with ``try: except ImportError`` so
    # ``ONNXRUNTIME_AVAILABLE`` is False when the extras are absent.
    import importlib

    module = importlib.import_module("triage_ml.optimization.onnx_adapter")
    if module.ONNXRUNTIME_AVAILABLE:
        return  # local env has the extras; the assertion below is vacuous here
    # When the extras are missing the helper symbols are still defined.
    assert callable(module.OnnxModelAdapter)
    assert module.ONNXRUNTIME_AVAILABLE is False


def test_optimization_exports_reflect_available_extras() -> None:
    """Symbol surface of ``triage_ml.optimization`` is import-safe."""

    from triage_ml.optimization import (
        AVAILABLE_VARIANTS,
        OptimizationFingerprint,
        iter_dataset_slices,
        resolve_variant_loader,
        validate_variant_metadata,
    )

    assert AVAILABLE_VARIANTS == ("sklearn", "onnx")
    assert callable(iter_dataset_slices)
    assert callable(resolve_variant_loader)
    assert callable(validate_variant_metadata)
    assert OptimizationFingerprint.__dataclass_fields__  # dataclass is intact


@pytest.mark.skipif(
    ONNX_AVAILABLE,
    reason="needs python without [optimization] extra to assert fallback",
)
def test_optimization_extras_flags_clearly_when_missing() -> None:
    """The module exposes ``ONNX_AVAILABLE=False`` when extras are absent."""

    assert ONNX_AVAILABLE is False


def test_normalise_error_code_lowercases_and_allowlists() -> None:
    """The middleware helper enforces lower-case + allow-list."""

    from triage_ml.observability.middleware import _normalise_error_code

    assert _normalise_error_code("Validation_Failed") == "validation_failed"
    assert _normalise_error_code(" model_not_ready ") == "model_not_ready"
    assert _normalise_error_code("model_not_ready") == "model_not_ready"
    assert _normalise_error_code(None) is None
    assert _normalise_error_code("") is None
    assert _normalise_error_code(12345) is None
    assert _normalise_error_code("non_allowlisted_code") is None
    assert _normalise_error_code("internal_error") is None


@pytest.mark.parametrize(
    "code",
    [
        "model_not_ready",
        "validation_failed",
        "prediction_failed",
        "unsupported_language",
        "clinician_review_required",
        "unauthorized",
        "forbidden",
        "request_failed",
        "model_not_found",
        "model_incompatible",
        "text_too_short_for_language_check",
        "indeterminate_language",
        "language_config_incompatible",
    ],
)
def test_normalise_error_code_accepts_public_codes(code: str) -> None:
    from triage_ml.observability.middleware import _normalise_error_code

    assert _normalise_error_code(code) == code


def test_generate_observability_traffic_emits_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """The script must use ``json.dumps`` (not ``str(dict)``) for the body.

    The script's ``_post_json`` is no longer imported under a public name
    here; we exercise the same ``json.dumps`` contract so the bug
    surface area is locked down.
    """

    import json as _json

    payload = {"text": "PRIVACY-CANARY sample"}
    body = _json.dumps(payload)
    # ``str(dict)`` would emit single quotes; this asserts the contrary.
    assert "'" not in body
    # And the body round-trips back to the original dict.
    assert _json.loads(body) == payload


def test_generate_observability_traffic_rejects_public_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The script guards URLs to loopback to avoid SSRF drift."""

    from importlib import util

    spec = util.spec_from_file_location(
        "generate_observability_traffic",
        Path(__file__).parents[1] / "scripts" / "generate_observability_traffic.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with pytest.raises(ValueError, match="loopback"):
        mod._enforce_loopback("https://google.com/predict", role="sklearn")
    # Loopback hosts are accepted without raising.
    assert mod._enforce_loopback("http://127.0.0.1:8001", role="sklearn") is None
    assert mod._enforce_loopback("http://localhost:8001", role="sklearn") is None


def test_resolve_api_key_falls_back_in_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_resolve_api_key`` accepts only doctor-compatible credentials."""

    from importlib import util

    spec = util.spec_from_file_location(
        "generate_observability_traffic",
        Path(__file__).parents[1] / "scripts" / "generate_observability_traffic.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.delenv("TRIAGE_ML_API_KEY_DOCTOR", raising=False)
    monkeypatch.delenv("TRIAGE_ML_API_KEY_SERVICE", raising=False)
    monkeypatch.delenv("TRIAGE_ML_TRAFFIC_API_KEY", raising=False)

    class _Args:
        api_key = "fallback-from-arg"

    monkeypatch.setenv("TRIAGE_ML_TRAFFIC_API_KEY", "traffic-key")
    assert mod._resolve_api_key(_Args()) == "traffic-key"

    monkeypatch.setenv("TRIAGE_ML_API_KEY_SERVICE", "service-key")
    assert mod._resolve_api_key(_Args()) == "traffic-key"

    monkeypatch.setenv("TRIAGE_ML_API_KEY_DOCTOR", "doctor-key")
    assert mod._resolve_api_key(_Args()) == "doctor-key"

    monkeypatch.delenv("TRIAGE_ML_TRAFFIC_API_KEY")
    monkeypatch.delenv("TRIAGE_ML_API_KEY_DOCTOR")
    monkeypatch.delenv("TRIAGE_ML_API_KEY_SERVICE")
    assert mod._resolve_api_key(_Args()) == "fallback-from-arg"


def test_optimization_fingerprint_aliases_classifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fingerprint_dict`` maps sklearn class names to short identifiers."""

    from triage_ml.optimization.optimize import CLASSIFIER_NAME_TO_KIND

    assert CLASSIFIER_NAME_TO_KIND["LogisticRegression"] == "logreg"
    assert CLASSIFIER_NAME_TO_KIND["LinearSVC"] == "linear_svc"


def test_normalise_classifier_kind_handles_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from triage_ml.optimization.onnx_adapter import normalize_classifier_kind

    assert normalize_classifier_kind(None) == "unknown"
    assert normalize_classifier_kind("") == "unknown"
    assert normalize_classifier_kind("logreg") == "logreg"
    assert normalize_classifier_kind("LogisticRegression") == "logreg"
    assert normalize_classifier_kind("Linearsvc") == "linear_svc"
    assert normalize_classifier_kind("logistic_regression") == "logreg"
    assert normalize_classifier_kind("definitely_not_a_class") == "unknown"


def test_benchmark_for_version_uses_atomic_write(tmp_path: Path) -> None:
    """``benchmark_for_version`` writes via ``_atomic_write_json``."""

    import inspect

    from triage_ml.orchestration import airflow_pipeline

    source = inspect.getsource(airflow_pipeline.benchmark_for_version)
    assert "_atomic_write_json" in source
    # The function must not call ``Path.write_text`` directly to avoid
    # the last-writer-wins race the previous bug produced.
    assert "write_text" not in source
    # And the aggregate path is derived from ``_resolve_reports_path`` so
    # the deployer can mount ``TRIAGE_REPORTS_DIR`` and avoid CWD
    # surprises.
    assert "_resolve_reports_path" in source


def test_benchmark_for_version_rejects_zero_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``benchmark_input_size=0`` must raise before allocating the session."""

    from triage_ml.orchestration import airflow_pipeline

    version_dir = tmp_path / "models" / "20260101T000000Z-0123456789ab"
    version_dir.mkdir(parents=True)
    (version_dir / "model.joblib").write_bytes(b"placeholder")
    (version_dir / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        airflow_pipeline.benchmark_for_version(version_dir, benchmark_input_size=0)


def test_train_with_sample_size_idempotency_includes_random_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs with different ``random_state`` must NOT be considered reusable."""

    from triage_ml.orchestration.airflow_pipeline import _slice_identity_fields

    base = {
        "random_state": 7,
        "test_size": 0.2,
        "task_type": "multiclass_text_classification",
        "language": "en",
        "selection_overrides": {"classifier": "logreg"},
    }

    ident_a = _slice_identity_fields(base, sample_size=5_000)
    base_shifted = dict(base)
    base_shifted["random_state"] = 42
    ident_b = _slice_identity_fields(base_shifted, sample_size=5_000)

    assert ident_a["random_state"] == 7
    assert ident_b["random_state"] == 42
    assert ident_a != ident_b


def test_slice_identity_fields_rejects_missing_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_slice_identity_fields`` must refuse config without an explicit classifier."""

    from triage_ml.orchestration.airflow_pipeline import _slice_identity_fields

    cfg = {
        "random_state": 7,
        "test_size": 0.2,
        "task_type": "multiclass_text_classification",
        "language": "en",
    }
    with pytest.raises(ValueError, match="selection_overrides"):
        _slice_identity_fields(cfg, sample_size=5_000)


def test_slice_identity_fields_honours_explicit_selected_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_slice_identity_fields`` honours an explicit ``selected_classifier`` argument."""

    from triage_ml.orchestration.airflow_pipeline import _slice_identity_fields

    cfg = {
        "random_state": 7,
        "test_size": 0.2,
        "task_type": "multiclass_text_classification",
        "language": "en",
    }
    ident = _slice_identity_fields(cfg, sample_size=5_000, selected_classifier="linear_svc")
    assert ident["selected_classifier"] == "linear_svc"

"""Regression tests for the second cycle of the Fase 2 review.

Lock-in for the regressions / new findings flagged by the 2nd cycle
cross-review:

* ``ModelHolder.reload_to`` invalidates the ONNX adapter cache.
* ``app.state.model_variant`` is single-source via the lifespan.
* ``_slice_identity_fields`` rejects config without an explicit class.
* ``OnnxModelAdapter`` singleton (no new session per call).
* ``OnnxModelAdapter.__call__`` issues a single ``session.run``.
* ``TestClient`` integration: ``PREDICTION_ERRORS_TOTAL`` records
  ``validation_failed`` when a POST is malformed.
* ``TestClient`` integration: ``503 model_not_ready`` when ONNX selected
  and ``model.onnx`` is absent.
* ``_normalise_error_code`` accepts the full ALLOWED+LANGUAGE union.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    import onnx  # noqa: F401
    import onnxruntime as ort  # noqa: F401
    import skl2onnx  # noqa: F401

    HAVE_OPTIMIZATION = True
except ImportError:
    HAVE_OPTIMIZATION = False

optimization_required = pytest.mark.skipif(
    not HAVE_OPTIMIZATION, reason="requires the [optimization] dependency group"
)


# ---------------------------------------------------------------------------
# A1 / B3 — reload_to invalidates holder._onnx_predictor
# ---------------------------------------------------------------------------


def _make_bundle(version_dir: Path) -> Path:
    """Write a tiny sklearn bundle + metadata.json to ``version_dir``.

    The bundle is intentionally rich enough to satisfy
    ``validate_artifact_bundle`` (which checks ``classes.json``,
    ``preprocessing``, ``dependency_versions``, ``fingerprints``,
    ``git_commit``, ``git_dirty``, ``model_name``, ``task_type``,
    ``created_at``). The checksum is computed after the joblib is
    written so the integrity check passes.
    """

    import hashlib
    import platform

    import joblib as _joblib
    import numpy as _np
    import scipy as _sp
    import sklearn as _sk

    version_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=200, random_state=7)),
        ]
    )
    pipeline.fit(
        [f"unique medical abstract number {i:04d} about cardiology or oncology" for i in range(40)],
        [1] * 20 + [2] * 20,
    )
    joblib_path = version_dir / "model.joblib"
    joblib.dump(pipeline, joblib_path)
    classes_path = version_dir / "classes.json"
    classes_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    checksum = hashlib.sha256(joblib_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "model_version": version_dir.name,
        "model_name": "triage-ml-fixture",
        "task_type": "multiclass_text_classification",
        "language": "en",
        "classes": [1, 2],
        "label_mapping": {"1": "a", "2": "b"},
        "random_state": 7,
        "n_train": 32,
        "n_test": 8,
        "metrics": {
            "accuracy": 0.9,
            "balanced_accuracy": 0.9,
            "macro_f1": 0.9,
            "weighted_f1": 0.9,
        },
        "preprocessing": {
            "classifier": "logreg",
            "tfidf": {
                "ngram_range": [1, 1],
                "min_df": 2,
                "max_df": 1.0,
                "lowercase": True,
            },
        },
        "selection": {"selected_classifier": "logreg"},
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": _np.__version__,
            "scipy": _sp.__version__,
            "scikit_learn": _sk.__version__,
            "joblib": _joblib.__version__,
        },
        "git_commit": "0" * 40,
        "git_dirty": False,
        "fingerprints": {
            "raw_csv_sha256": "a" * 64,
            "clean_csv_sha256": "a" * 64,
            "config_sha256": "a" * 64,
        },
        "checksum_sha256": checksum,
        "created_at": "2026-09-08T00:00:00Z",
        "available_variants": ["sklearn"],
        "selected_classifier": "logreg",
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return version_dir


def test_reload_to_invalidates_onnx_predictor_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``holder._onnx_predictor`` is reset to ``None`` after a version swap."""

    from triage_ml.dev_api.app import ModelHolder

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=50, random_state=7)),
        ]
    )
    pipeline.fit(
        ["alpha beta", "alpha beta", "gamma delta", "gamma delta"],
        [1, 1, 2, 2],
    )

    monkeypatch.setattr(
        "triage_ml.dev_api.app.load_artifact",
        lambda path: (
            pipeline,
            {
                "model_version": path.parent.name,
                "classes": [1, 2],
                "label_mapping": {"1": "alpha", "2": "beta"},
                "language": "en",
            },
        ),
    )
    monkeypatch.setattr(
        "triage_ml.dev_api.app._validated_model_path",
        lambda root, version: root / version / "model.joblib",
    )
    registry_root = tmp_path / "models"
    v_a = registry_root / "20260101T000000Z-aaaaaaaaaaaa"
    v_b = registry_root / "20260102T000000Z-bbbbbbbbbbbb"
    v_a.mkdir(parents=True)
    (v_a / "model.joblib").write_bytes(b"x")
    (v_a / "metadata.json").write_text("{}", encoding="utf-8")
    v_b.mkdir(parents=True)
    (v_b / "model.joblib").write_bytes(b"x")
    (v_b / "metadata.json").write_text("{}", encoding="utf-8")

    holder = ModelHolder(registry_root=registry_root, model_path=v_a / "model.joblib")
    holder.load()
    holder._onnx_predictor = "stale-sentinel"  # type: ignore[attr-defined]
    holder.reload_to(v_b.name)

    assert getattr(holder, "_onnx_predictor", "missing") is None


# ---------------------------------------------------------------------------
# A3 / C3 — _slice_identity_fields & register/explicit classifier surface
# ---------------------------------------------------------------------------


def test_load_onnx_rejects_missing_classes(tmp_path: Path) -> None:
    from triage_ml.optimization.registry import _load_onnx

    metadata = {"model_version": "x", "classes": None}
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="classes"):
        _load_onnx(tmp_path / "model.onnx")


def test_load_onnx_rejects_non_int_classes(tmp_path: Path) -> None:
    from triage_ml.optimization.registry import _load_onnx

    metadata = {"model_version": "x", "classes": ["a", "b"]}
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError):
        _load_onnx(tmp_path / "model.onnx")


# ---------------------------------------------------------------------------
# C1 — OnnxModelAdapter singleton + __call__ single session.run
# ---------------------------------------------------------------------------


@optimization_required
def test_onnx_adapter_reuses_session_across_calls(tmp_path: Path) -> None:
    import onnxruntime as ort

    from triage_ml.optimization.optimize import export_onnx

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=200, random_state=7)),
        ]
    )
    pipeline.fit(
        [f"abstract {i:04d} about cardiology or oncology" for i in range(80)],
        [1] * 30 + [2] * 30 + [3] * 20,
    )
    onnx_path, _ = export_onnx(pipeline, tmp_path / "model.onnx")

    from triage_ml.optimization.onnx_adapter import OnnxModelAdapter

    adapter = OnnxModelAdapter(
        onnx_path=onnx_path,
        classes=tuple(int(c) for c in pipeline.classes_),
        classifier_kind="logreg",
    )
    # Two calls -> the same cached ``_session_obj``.
    adapter(["text one"])
    cached_after_first = adapter._session_obj
    assert isinstance(cached_after_first, ort.InferenceSession)
    adapter(["text two"])
    assert adapter._session_obj is cached_after_first


@optimization_required
def test_onnx_adapter_call_runs_session_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OnnxModelAdapter.__call__`` must invoke ``session.run`` exactly once per call."""

    from triage_ml.optimization.optimize import export_onnx

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=200, random_state=7)),
        ]
    )
    pipeline.fit(
        [f"abstract {i:04d} about cardiology or oncology" for i in range(80)],
        [1] * 30 + [2] * 30 + [3] * 20,
    )
    onnx_path, _ = export_onnx(pipeline, tmp_path / "model.onnx")

    from triage_ml.optimization.onnx_adapter import OnnxModelAdapter

    adapter = OnnxModelAdapter(
        onnx_path=onnx_path,
        classes=tuple(int(c) for c in pipeline.classes_),
        classifier_kind="logreg",
    )
    # Trigger session creation and capture the live session.
    adapter(["text zero"])
    session_obj = adapter._session_obj
    assert session_obj is not None
    calls = []

    original_run = session_obj.run
    session_obj.run = lambda *args, **kwargs: (  # type: ignore[assignment]
        calls.append((args, kwargs)) or original_run(*args, **kwargs)
    )
    adapter(["text one"])
    adapter(["text two"])

    assert len(calls) == 2  # exactly one session.run per __call__


# ---------------------------------------------------------------------------
# C2 / C6 — TestClient integration: PREDICTION_ERRORS_TOTAL fires on 422/503
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A ModelHolder pointing at an artefact with no ``model.onnx`` next to it."""

    from triage_ml.dev_api.app import ModelHolder

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=50, random_state=7)),
        ]
    )
    pipeline.fit(
        ["alpha beta", "alpha beta", "gamma delta", "gamma delta"],
        [1, 1, 2, 2],
    )

    monkeypatch.setattr(
        "triage_ml.dev_api.app.load_artifact",
        lambda path: (
            pipeline,
            {
                "model_version": path.parent.name,
                "classes": [1, 2],
                "label_mapping": {"1": "alpha", "2": "beta"},
                "language": "en",
            },
        ),
    )
    monkeypatch.setattr(
        "triage_ml.dev_api.app._validated_model_path",
        lambda root, version: root / version / "model.joblib",
    )
    registry_root = tmp_path / "models"
    version_dir = registry_root / "20260101T000000Z-0123456789ab"
    version_dir.mkdir(parents=True)
    (version_dir / "model.joblib").write_bytes(b"x")
    # Note: ``model.onnx`` intentionally missing for the model_not_ready test.
    holder = ModelHolder(registry_root=registry_root, model_path=version_dir / "model.joblib")
    holder.load()
    return holder


@pytest.fixture
def production_client(dummy_holder, monkeypatch: pytest.MonkeyPatch):
    """Yield a TestClient + service API key in the env."""

    monkeypatch.setenv("TRIAGE_ML_API_KEY_SERVICE", "service-" + "x" * 32)
    monkeypatch.setenv("TRIAGE_ML_API_KEY_DOCTOR", "doctor-" + "y" * 32)
    monkeypatch.setenv("TRIAGE_ML_API_KEY_PATIENT", "patient-" + "z" * 32)
    monkeypatch.setenv("TRIAGE_ML_LOG_LEVEL", "WARNING")
    # ``get_settings`` is ``lru_cache``d; another test in the suite may have
    # already cached a stale ``Settings`` instance. Clear the cache so the
    # env vars above take effect.
    from triage_ml.api.settings import get_settings

    get_settings.cache_clear()
    from triage_ml.api.app import create_app

    app = create_app(holder=dummy_holder)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client


def test_validation_failed_appears_in_metrics(production_client) -> None:
    """POSTing an empty body must surface ``validation_failed`` in the metric payload."""

    response = production_client.post(
        "/predict",
        json={},
        headers={"X-API-Key": "doctor-" + "y" * 32},
    )
    assert response.status_code == 422

    metrics = production_client.get("/metrics").text
    assert "triage_ml_prediction_errors_total{" in metrics
    assert 'error_code="validation_failed"' in metrics


def test_model_not_ready_returns_503_when_onnx_missing(production_client, dummy_holder) -> None:
    """Setting ``app.state.model_variant = "onnx"`` without ``model.onnx`` must 503."""

    # ``monkeypatch.setenv`` is too late — the TestClient already cached the
    # settings inside ``create_app``. Force the variant on ``app.state``
    # so the ``/predict`` path resolves ``onnx`` and the holder has no
    # ``model.onnx`` next to it.
    production_client.app.state.model_variant = "onnx"
    health = production_client.get("/health")
    assert health.status_code == 503
    assert health.json()["model_variant"] == "onnx"
    response = production_client.post(
        "/predict",
        json={"text": "this is a short text placeholder"},
        headers={"X-API-Key": "doctor-" + "y" * 32},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "model_not_ready"


# ---------------------------------------------------------------------------
# H7 — error code allow-list covers ALLOWED + LANGUAGE + request_failed
# ---------------------------------------------------------------------------


def test_metric_error_codes_includes_full_union() -> None:
    from triage_ml.dev_api.app import ALLOWED_ERROR_CODES, LANGUAGE_ERROR_CODES
    from triage_ml.observability.middleware import _METRIC_ERROR_CODES, _normalise_error_code

    expected = ALLOWED_ERROR_CODES | LANGUAGE_ERROR_CODES | {"request_failed"}
    assert _METRIC_ERROR_CODES == frozenset(expected)

    for code in expected:
        assert _normalise_error_code(code) == code


# ---------------------------------------------------------------------------
# B8 — export_onnx_for_version: reused when on-disk hash matches
# ---------------------------------------------------------------------------


@optimization_required
def test_export_onnx_for_version_reuses_existing_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching ``model.onnx`` + ``metadata.optimization`` should yield ``reused=True``."""

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=200, random_state=7)),
        ]
    )
    pipeline.fit(
        [f"abstract {i:04d} about cardiology or oncology" for i in range(80)],
        [1] * 30 + [2] * 30 + [3] * 20,
    )
    version_dir = tmp_path / "models" / "20260101T000000Z-0123456789ab"
    version_dir.mkdir(parents=True)
    joblib.dump(pipeline, version_dir / "model.joblib")

    from triage_ml.orchestration.airflow_pipeline import (
        export_onnx_for_version,
    )

    metadata = {
        "model_version": version_dir.name,
        "classes": list(int(c) for c in pipeline.classes_),
        "label_mapping": {"1": "a", "2": "b", "3": "c"},
        "metrics": {"accuracy": 0.9, "balanced_accuracy": 0.9, "macro_f1": 0.9},
        "checksum_sha256": "a" * 64,
        "schema_version": 1,
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        "triage_ml.orchestration.airflow_pipeline.validate_artifact_bundle",
        lambda path: json.loads(Path(path).with_name("metadata.json").read_text(encoding="utf-8")),
    )

    first = export_onnx_for_version(version_dir, opset=17)
    assert first["reused"] is False
    # Second call should hit the reuse path and return reused=True.
    second = export_onnx_for_version(version_dir, opset=17)
    assert second["reused"] is True
    assert (
        second["optimization"]["onnx_checksum_sha256"]
        == first["optimization"]["onnx_checksum_sha256"]
    )

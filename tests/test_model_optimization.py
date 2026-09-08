"""Tests for ``triage_ml.optimization.optimize`` and the ONNX adapter — Etapa 5.

These tests require the optional ``[optimization]`` dependency group (skl2onnx,
onnxruntime, onnx) to be installed. When the extras are missing the tests
auto-skip via the ``optimization_required`` module-level marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import onnx  # noqa: F401
    import onnxruntime as ort  # noqa: F401
    import skl2onnx  # noqa: F401

    HAVE_OPTIMIZATION = True
except ImportError:  # pragma: no cover - exercised when extras missing
    HAVE_OPTIMIZATION = False

optimization_required = pytest.mark.skipif(
    not HAVE_OPTIMIZATION, reason="requires the [optimization] dependency group"
)


pytestmark = optimization_required


@pytest.fixture
def fitted_pipeline() -> object:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(0)
    texts = [
        f"unique medical abstract number {i:04d} about cardiology or oncology" for i in range(200)
    ]
    labels = rng.integers(1, 4, size=200)

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1), lowercase=True)),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def test_export_onnx_writes_file_and_fingerprint(tmp_path: Path, fitted_pipeline) -> None:
    from triage_ml.optimization.optimize import (
        OptimizationFingerprint,
        export_onnx,
        fingerprint_hash,
    )

    out_path, fingerprint = export_onnx(fitted_pipeline, tmp_path / "model.onnx")

    assert out_path.is_file()
    assert out_path.stat().st_size > 0
    assert isinstance(fingerprint, OptimizationFingerprint)
    assert fingerprint.opset == 17
    assert fingerprint.classifier == "LogisticRegression"
    assert fingerprint.quantized is False
    assert len(fingerprint_hash(fingerprint)) == 16


def test_export_onnx_rejects_wrong_pipeline_shape(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression

    from triage_ml.optimization.optimize import export_onnx

    bogus = LogisticRegression()
    with pytest.raises(ValueError):
        export_onnx(bogus, tmp_path / "model.onnx")


def test_export_onnx_supports_custom_opset(tmp_path: Path, fitted_pipeline) -> None:
    from triage_ml.optimization.optimize import export_onnx

    out_path, fingerprint = export_onnx(fitted_pipeline, tmp_path / "model.onnx", opset=18)
    assert out_path.is_file()
    assert fingerprint.opset == 18


def test_onnx_adapter_predict_matches_sklearn(tmp_path: Path, fitted_pipeline) -> None:
    """The ONNX adapter must reproduce sklearn on the same inputs."""

    from triage_ml.optimization.onnx_adapter import OnnxModelAdapter
    from triage_ml.optimization.optimize import export_onnx

    out_path, _ = export_onnx(fitted_pipeline, tmp_path / "model.onnx")
    adapter = OnnxModelAdapter(
        onnx_path=out_path,
        classes=tuple(int(label) for label in fitted_pipeline.classes_),
        classifier_kind="logreg",
    )

    inputs = ["cardiology abstract about heart", "oncology abstract about tumour"]
    sklearn_predictions = [int(label) for label in fitted_pipeline.predict(inputs)]
    onnx_predictions = [int(label) for label in adapter.predict(inputs)]

    assert onnx_predictions == sklearn_predictions


def test_onnx_adapter_score_for_logreg_returns_proba(tmp_path: Path, fitted_pipeline) -> None:
    from triage_ml.optimization.onnx_adapter import OnnxModelAdapter
    from triage_ml.optimization.optimize import export_onnx

    out_path, _ = export_onnx(fitted_pipeline, tmp_path / "model.onnx")
    adapter = OnnxModelAdapter(
        onnx_path=out_path,
        classes=tuple(int(label) for label in fitted_pipeline.classes_),
        classifier_kind="logreg",
    )

    score, kind = adapter.score_for(["cardiology abstract"], predicted_label=adapter.classes[0])
    assert kind == "predict_proba"
    assert score is not None and 0.0 <= score <= 1.0


def test_onnx_adapter_predict_proba_returns_none_for_linear_svc(
    tmp_path: Path, fitted_pipeline
) -> None:
    """LinearSVC has no probability surface; the adapter must report that."""

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    from triage_ml.optimization.onnx_adapter import OnnxModelAdapter
    from triage_ml.optimization.optimize import export_onnx

    rng = np.random.default_rng(0)
    texts = [
        f"unique medical abstract number {i:04d} about cardiology or oncology" for i in range(200)
    ]
    labels = rng.integers(1, 4, size=200)
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 1), lowercase=True)),
            ("clf", LinearSVC()),
        ]
    )
    pipeline.fit(texts, labels)

    out_path, _ = export_onnx(pipeline, tmp_path / "model.onnx")
    adapter = OnnxModelAdapter(
        onnx_path=out_path,
        classes=tuple(int(label) for label in pipeline.classes_),
        classifier_kind="linear_svc",
    )

    assert adapter.predict_proba(["any text here"]) is None
    score, kind = adapter.score_for(["any text here"], predicted_label=adapter.classes[0])
    assert kind in {"decision_function", "absent"}

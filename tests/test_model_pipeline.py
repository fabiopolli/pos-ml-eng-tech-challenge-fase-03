"""Tests for the ``triage_ml.models.pipeline`` factory."""

from __future__ import annotations

import numpy as np
import pytest

from triage_ml.models.pipeline import (
    DEFAULT_LINEAR_SVC,
    DEFAULT_LOGREG,
    DEFAULT_TFIDF,
    VALID_CLASSIFIERS,
    build_classifier,
    build_pipeline,
)


def test_build_pipeline_returns_expected_steps() -> None:
    pipeline = build_pipeline()
    assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]


def test_build_pipeline_logreg_exposes_predict_proba() -> None:
    pipeline = build_pipeline("logreg")
    assert hasattr(pipeline.named_steps["clf"], "predict_proba")


def test_build_pipeline_uses_tfidf_defaults() -> None:
    pipeline = build_pipeline("logreg")
    tfidf = pipeline.named_steps["tfidf"]
    assert tfidf.ngram_range == DEFAULT_TFIDF["ngram_range"]
    assert tfidf.min_df == DEFAULT_TFIDF["min_df"]
    assert tfidf.max_df == DEFAULT_TFIDF["max_df"]
    assert tfidf.sublinear_tf is True


def test_build_classifier_logreg_defaults() -> None:
    clf = build_classifier("logreg")
    assert clf.get_params()["solver"] == DEFAULT_LOGREG["solver"]
    assert clf.get_params()["max_iter"] == DEFAULT_LOGREG["max_iter"]
    assert clf.get_params()["class_weight"] == DEFAULT_LOGREG["class_weight"]


def test_build_classifier_overrides_apply() -> None:
    clf = build_classifier("linear_svc", C=2.5)
    assert clf.get_params()["C"] == 2.5
    assert clf.get_params()["class_weight"] == DEFAULT_LINEAR_SVC["class_weight"]


def test_build_pipeline_rejects_unknown_classifier() -> None:
    with pytest.raises(ValueError):
        build_pipeline("not-a-classifier")


def test_build_classifier_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_classifier("random_forest")


def test_pipeline_end_to_end_on_tiny_corpus() -> None:
    pipeline = build_pipeline("logreg")
    texts = [
        "tumor growth in the liver of the patient",
        "cardiac arrest after exercise",
        "brain tumor removed successfully",
        "heart attack in elderly patient",
        "liver metastases detected",
    ]
    labels = [1, 4, 3, 4, 1]
    pipeline.fit(texts, labels)
    preds = pipeline.predict(texts[:2])
    assert isinstance(preds, np.ndarray)
    assert set(preds.tolist()).issubset(set(labels))


def test_valid_classifiers_constant() -> None:
    assert set(VALID_CLASSIFIERS) == {"logreg", "linear_svc"}

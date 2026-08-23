"""Factory for the baseline text classification pipeline.

The pipeline wraps a ``TfidfVectorizer`` and a linear classifier into a
single scikit-learn ``Pipeline`` so that it can be persisted with
``joblib`` and consumed by the API without leaking preprocessing
details.

Two classifiers are supported:

- ``"logreg"`` (default): ``LogisticRegression`` with
  ``class_weight="balanced"`` and the ``lbfgs`` solver. It supports
  multiclass natively and exposes ``predict_proba`` for the API's
  ``score`` field.
- ``"linear_svc"``: ``LinearSVC`` with ``class_weight="balanced"``.
  Slightly faster at inference but does not provide calibrated
  probabilities out of the box, so ``score`` is reported as ``None``
  unless a calibrated wrapper is configured.

The hyperparameters are loaded from ``configs/training.yaml`` by
``triage_ml.models.train.run_training`` and passed in through ``tfidf``
and ``clf`` keyword arguments. Defaults here are conservative values
that mirror the YAML file and keep ``build_pipeline()`` working
without configuration, which is what the smoke tests rely on.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

VALID_CLASSIFIERS = ("logreg", "linear_svc")

DEFAULT_TFIDF: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "lowercase": True,
    "token_pattern": r"(?u)\b\w+\b",
}

DEFAULT_LOGREG: dict[str, Any] = {
    "solver": "lbfgs",
    "max_iter": 2000,
    "class_weight": "balanced",
    "C": 1.0,
    "random_state": 42,
}

DEFAULT_LINEAR_SVC: dict[str, Any] = {
    "class_weight": "balanced",
    "C": 1.0,
    "random_state": 42,
}


def build_classifier(name: str, **overrides: Any) -> Any:
    """Build the classifier step by name.

    Unknown overrides are applied on top of the defaults for the chosen
    classifier, making it easy to experiment without rewriting this
    factory.
    """

    if name not in VALID_CLASSIFIERS:
        raise ValueError(f"Unknown classifier {name!r}. Valid options: {VALID_CLASSIFIERS}")

    if name == "logreg":
        params = {**DEFAULT_LOGREG, **overrides}
        return LogisticRegression(**params)
    params = {**DEFAULT_LINEAR_SVC, **overrides}
    return LinearSVC(**params)


def build_pipeline(
    classifier: str = "logreg",
    *,
    tfidf: dict[str, Any] | None = None,
    classifier_kwargs: dict[str, Any] | None = None,
) -> Pipeline:
    """Return a fresh, unfitted TF-IDF + linear classifier pipeline."""

    tfidf_params = {**DEFAULT_TFIDF, **(tfidf or {})}
    clf = build_classifier(classifier, **(classifier_kwargs or {}))
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", clf),
        ]
    )

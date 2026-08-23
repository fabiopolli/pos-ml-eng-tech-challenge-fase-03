"""Hermetic tests for the language detection layer and its HTTP wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import joblib
import pytest
from fastapi.testclient import TestClient

from triage_ml.dev_api import language as api_language
from triage_ml.dev_api.app import ModelHolder, create_app
from triage_ml.dev_api.config import ApiConfig, reset_api_config_cache
from triage_ml.dev_api.language import UnsupportedLanguageError, detect_language
from triage_ml.models.artifact import ArtifactPaths, build_metadata, write_classes, write_metadata
from triage_ml.models.pipeline import build_pipeline

VERSION = "20260823T120000Z-0123456789ab"
DIGEST = "a" * 64
EN_TEXT = (
    "We report a 62-year-old patient with acute myocardial infarction after chest pain. "
    "Coronary angiography confirmed occlusion of the left anterior descending artery."
)
PT_TEXT = (
    "Relatamos um paciente de 62 anos com infarto agudo do miocárdio após dor torácica. "
    "A angiocoronariografia confirmou oclusão da artéria descendente anterior esquerda."
)


def _artifact(tmp_path: Path) -> Path:
    pipeline = build_pipeline(
        "logreg",
        tfidf={"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
    )
    texts = [
        "liver tumor neoplasm",
        "digestive stomach disease",
        "brain nervous disorder",
        "heart cardiovascular disease",
        "general fever condition",
    ] * 2
    labels = [1, 2, 3, 4, 5] * 2
    pipeline.fit(texts, labels)
    paths = ArtifactPaths.for_version(tmp_path, VERSION)
    paths.ensure()
    joblib.dump(pipeline, paths.joblib)
    write_classes(paths.classes, pipeline.classes_)
    metadata = build_metadata(
        model_version=VERSION,
        model_name="tiny",
        task_type="multiclass_text_classification",
        language="en",
        classes=list(pipeline.classes_),
        label_mapping={
            "1": "neoplasms",
            "2": "digestive system diseases",
            "3": "nervous system diseases",
            "4": "cardiovascular diseases",
            "5": "general pathological conditions",
        },
        random_state=42,
        n_train=10,
        n_test=5,
        metrics={
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
            "per_class": {
                str(label): {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "support": 1,
                }
                for label in range(1, 6)
            },
        },
        preprocessing={
            "vectorizer": "tfidf",
            "tfidf": {},
            "classifier": "logreg",
            "classifier_params": {},
        },
        selection={
            "metric": "macro_f1",
            "folds": 2,
            "candidates": {
                name: {
                    "fold_macro_f1": [1.0, 1.0],
                    "mean_macro_f1": 1.0,
                    "std_macro_f1": 0.0,
                }
                for name in ("logreg", "linear_svc")
            },
            "selected_classifier": "logreg",
            "test_set_used_for_selection": False,
        },
        dependency_versions={
            "python": "3.12",
            "numpy": "2.0",
            "scipy": "1.0",
            "scikit_learn": "1.0",
            "joblib": "1.0",
        },
        git_commit="0" * 40,
        git_dirty=False,
        fingerprints={
            "raw_csv_sha256": DIGEST,
            "prepared_dataset_sha256": DIGEST,
            "train_split_sha256": DIGEST,
            "test_split_sha256": DIGEST,
            "config_sha256": DIGEST,
        },
        joblib_path=paths.joblib,
    )
    write_metadata(paths.metadata, metadata)
    return paths.joblib


@pytest.fixture()
def holder(tmp_path: Path) -> ModelHolder:
    return ModelHolder(_artifact(tmp_path))


@pytest.fixture()
def client(holder: ModelHolder):
    with TestClient(create_app(holder=holder)) as test_client:
        yield test_client


@pytest.fixture()
def fixed_config(monkeypatch: pytest.MonkeyPatch) -> ApiConfig:
    """Pin the API config to deterministic thresholds for tests."""

    config = ApiConfig(
        supported_languages={"en"},
        min_text_chars_for_language_check=20,
        min_language_score=0.85,
    )
    monkeypatch.setattr("triage_ml.dev_api.app.get_api_config", lambda: config)
    reset_api_config_cache()
    yield config
    reset_api_config_cache()


def test_detect_language_accepts_supported_language() -> None:
    check = detect_language(EN_TEXT, min_chars=20, min_score=0.0, supported={"en"})
    assert check.accepted is True
    assert check.code == "en"
    assert check.score is not None and 0.0 <= check.score <= 1.0


def test_detect_language_rejects_short_text() -> None:
    with pytest.raises(UnsupportedLanguageError) as exc_info:
        detect_language("short", min_chars=20, min_score=0.0, supported={"en"})
    assert exc_info.value.reason == "text_too_short_for_language_check"
    assert exc_info.value.code is None
    assert exc_info.value.score is None


def test_detect_language_rejects_low_confidence() -> None:
    with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("en", 0.1)):
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            detect_language(EN_TEXT, min_chars=20, min_score=0.85, supported={"en"})
    assert exc_info.value.reason == "indeterminate_language"
    assert exc_info.value.code == "en"
    assert exc_info.value.score == pytest.approx(0.1)


def test_detect_language_rejects_unsupported_language() -> None:
    # Force the verdict to be Portuguese at a confident score
    with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("pt", 0.9)):
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            detect_language(PT_TEXT, min_chars=20, min_score=0.0, supported={"en"})
    assert exc_info.value.reason == "unsupported_language"
    assert exc_info.value.code == "pt"
    assert exc_info.value.score == pytest.approx(0.9)


def test_api_predict_accepts_english_text(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """English input goes through detection and prediction cleanly."""

    config = ApiConfig(
        supported_languages={"en"},
        min_text_chars_for_language_check=20,
        min_language_score=0.0,
    )
    monkeypatch.setattr("triage_ml.dev_api.app.get_api_config", lambda: config)
    reset_api_config_cache()
    try:
        response = client.post("/predict", json={"text": EN_TEXT})
        assert response.status_code == 200
        body = response.json()
        assert body["label_name"] in {
            "neoplasms",
            "digestive system diseases",
            "nervous system diseases",
            "cardiovascular diseases",
            "general pathological conditions",
        }
        timing = response.headers["server-timing"]
        assert "detect;dur=" in timing
        assert "predict;dur=" in timing
    finally:
        reset_api_config_cache()


def test_api_predict_rejects_non_english_text(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-English input is rejected without going through the model."""

    from triage_ml.dev_api.config import ApiConfig

    config = ApiConfig(
        supported_languages={"en"},
        min_text_chars_for_language_check=20,
        min_language_score=0.0,
    )
    monkeypatch.setattr("triage_ml.dev_api.app.get_api_config", lambda: config)
    reset_api_config_cache()
    try:
        with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("pt", 0.9)):
            response = client.post("/predict", json={"text": PT_TEXT})
        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == "unsupported_language"
        assert body["detected_language"] == "pt"
        assert body["detected_language_score"] == pytest.approx(0.9)
        assert PT_TEXT not in response.text
        timing = response.headers["server-timing"]
        assert "detect;dur=" in timing
        assert "predict;dur=" not in timing
    finally:
        reset_api_config_cache()


def test_api_predict_rejects_short_text_even_in_english(
    client: TestClient,
    fixed_config: ApiConfig,
) -> None:
    response = client.post("/predict", json={"text": "liver tumor"})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "text_too_short_for_language_check"
    assert body["detected_language"] is None
    assert body["detected_language_score"] is None


def test_api_predict_rejects_indeterminate_language(
    client: TestClient,
    fixed_config: ApiConfig,
) -> None:
    with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("en", 0.1)):
        response = client.post("/predict", json={"text": EN_TEXT})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "indeterminate_language"
    assert body["detected_language"] == "en"
    assert body["detected_language_score"] == pytest.approx(0.1)


def test_api_predict_language_error_does_not_leak_text_in_logs(
    client: TestClient,
    fixed_config: ApiConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "TEXTO_CLINICO_SENTINELA_PORTUGUES"
    payload = (
        f"{sentinel} - relatamos um paciente com dor torácica aguda e dispneia progressiva, "
        "submetido a angiocoronariografia que confirmou oclusão arterial importante."
    )
    with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("pt", 0.9)):
        with caplog.at_level("INFO", logger="triage_ml.dev_api"):
            response = client.post("/predict", json={"text": payload})
    assert response.status_code == 422
    assert sentinel not in response.text
    assert sentinel not in caplog.text
    assert "rid=" in caplog.text  # request id is logged, never the body


def test_api_predict_uses_configured_thresholds(
    client: TestClient,
    fixed_config: ApiConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom thresholds from ApiConfig are honoured at request time."""

    permissive = ApiConfig(
        supported_languages={"en"},
        min_text_chars_for_language_check=5,
        min_language_score=0.8,
    )
    monkeypatch.setattr("triage_ml.dev_api.app.get_api_config", lambda: permissive)
    reset_api_config_cache()
    try:
        with patch.object(api_language.LANGUAGE_IDENTIFIER, "classify", return_value=("en", 0.9)):
            response = client.post("/predict", json={"text": EN_TEXT})
        assert response.status_code == 200
        timing = response.headers["server-timing"]
        assert "predict;dur=" in timing
    finally:
        reset_api_config_cache()

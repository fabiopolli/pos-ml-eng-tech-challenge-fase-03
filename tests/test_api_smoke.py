"""Hermetic contract tests for the local smoke API."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from triage_ml.api.app import ModelHolder, create_app
from triage_ml.models.artifact import ArtifactPaths, build_metadata, write_classes, write_metadata
from triage_ml.models.pipeline import build_pipeline

VERSION = "20260823T120000Z-0123456789ab"
DIGEST = "a" * 64


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


def test_health_returns_validated_model_metadata(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": VERSION,
        "model_loaded": True,
    }
    assert response.headers["x-request-id"]


def test_predict_uses_manifest_mapping_and_emits_timing(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "liver tumor neoplasm"})
    assert response.status_code == 200
    body = response.json()
    assert body["label_name"] in {
        "neoplasms",
        "digestive system diseases",
        "nervous system diseases",
        "cardiovascular diseases",
        "general pathological conditions",
    }
    assert body["score"] is None or 0.0 <= body["score"] <= 1.0
    assert body["latency_ms"] >= 0
    assert response.headers["server-timing"].startswith("predict;dur=")
    header_ms = float(response.headers["server-timing"].split("=")[1])
    assert header_ms == pytest.approx(body["latency_ms"], abs=0.001)
    assert response.headers["x-request-id"] == body["request_id"]


def test_client_request_id_is_never_trusted(
    client: TestClient,
) -> None:
    client_supplied = "UNIQUE_CLINICAL_SENTINEL"
    response = client.post(
        "/predict",
        json={"text": "heart disease"},
        headers={"X-Request-ID": client_supplied},
    )
    assert response.headers["x-request-id"] == response.json()["request_id"]
    assert response.json()["request_id"] != client_supplied


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   \n\t"}, {}])
def test_invalid_text_returns_sanitized_422(client: TestClient, payload: dict[str, str]) -> None:
    sentinel = payload.get("text", "missing-sentinel")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "validation_failed"
    assert body["message"] in {"Request body is invalid.", "Request could not be processed."}
    if sentinel:
        assert sentinel not in response.text
    assert response.headers["x-request-id"] == body["request_id"]


def test_padding_is_stripped_by_request_schema(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "   liver tumor   "})
    assert response.status_code == 200
    assert response.json()["warnings"] == []


def test_prediction_failure_is_sanitized_and_does_not_log_text(
    client: TestClient,
    holder: ModelHolder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "UNIQUE_CLINICAL_SENTINEL"

    class FailingPipeline:
        def predict(self, _: list[str]) -> None:
            raise RuntimeError(f"failed for {sentinel}")

    holder.pipeline = FailingPipeline()
    with caplog.at_level("ERROR", logger="triage_ml.api"):
        response = client.post("/predict", json={"text": sentinel})
    assert response.status_code == 500
    assert response.json()["error_code"] == "prediction_failed"
    assert sentinel not in response.text
    assert sentinel not in caplog.text
    assert response.headers["server-timing"].startswith("predict;dur=")


def test_invalid_artifact_fails_during_startup(tmp_path: Path) -> None:
    model_path = _artifact(tmp_path)
    model_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="missing or incompatible"):
        with TestClient(create_app(model_path=model_path)):
            pass

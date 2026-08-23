"""Hermetic contract tests for the local dev API."""

from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import scipy
import sklearn
from fastapi.testclient import TestClient

from triage_ml.dev_api.app import ModelHolder, create_app
from triage_ml.models.artifact import ArtifactPaths, build_metadata, write_classes, write_metadata
from triage_ml.models.pipeline import build_pipeline

VERSION = "20260823T120000Z-0123456789ab"
DIGEST = "a" * 64


def _artifact(tmp_path: Path, *, version: str = VERSION) -> Path:
    """Write a single validated artifact under ``tmp_path/<version>/model.joblib``.

    ``tmp_path`` is the parent directory of the version directory — i.e.
    ``tmp_path`` should be ``<repo>/models`` when constructing a fixture
    for the dashboard picker. The default ``version`` matches the
    top-level ``VERSION`` constant so existing callers stay compatible.
    """

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
    paths = ArtifactPaths.for_version(tmp_path, version)
    paths.ensure()
    joblib.dump(pipeline, paths.joblib)
    write_classes(paths.classes, pipeline.classes_)
    metadata = build_metadata(
        model_version=version,
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
            "best_classifier": "logreg",
            "selected_classifier": "logreg",
            "selection_policy": "highest_mean_macro_f1",
            "test_set_used_for_selection": False,
        },
        dependency_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
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
    response = client.post(
        "/predict",
        json={
            "text": (
                "We report a 62-year-old patient with an aggressive liver tumor that "
                "required urgent surgical resection and histopathological evaluation."
            )
        },
    )
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
    timing = response.headers["server-timing"]
    assert "detect;dur=" in timing
    assert "predict;dur=" in timing
    predict_part = next(p for p in timing.split(", ") if p.startswith("predict;dur="))
    header_ms = float(predict_part.split("=")[1])
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


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   \n\t"}, {"text": "x" * 20_001}, {}])
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
    response = client.post(
        "/predict",
        json={
            "text": "   "
            "We report a 62-year-old patient with an aggressive liver tumor requiring "
            "urgent surgical resection and histopathological evaluation.   "
        },
    )
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
    with caplog.at_level("ERROR", logger="triage_ml.dev_api"):
        response = client.post(
            "/predict",
            json={
                "text": (
                    f"{sentinel} - we report a 62-year-old patient presenting with chest "
                    "pain and ST-segment elevation requiring urgent cardiac catheterization."
                )
            },
        )
    assert response.status_code == 500
    assert response.json()["error_code"] == "prediction_failed"
    assert sentinel not in response.text
    assert sentinel not in caplog.text
    timing = response.headers["server-timing"]
    assert "detect;dur=" in timing
    assert "predict;dur=" in timing


def test_invalid_artifact_fails_during_startup(tmp_path: Path) -> None:
    model_path = _artifact(tmp_path)
    model_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="missing or incompatible"):
        with TestClient(create_app(model_path=model_path)):
            pass


def test_model_info_returns_validated_manifest(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    # Top-level identity fields surfaced from metadata.json.
    assert body["model_version"] == VERSION
    assert body["model_name"] == "tiny"
    assert body["task_type"] == "multiclass_text_classification"
    assert body["language"] == "en"
    assert body["classes"] == [1, 2, 3, 4, 5]
    assert body["label_mapping"]["1"] == "neoplasms"
    assert body["n_train"] == 10
    assert body["n_test"] == 5
    # Metrics block is preserved (global + per_class).
    assert body["metrics"]["macro_f1"] == 1.0
    assert set(body["metrics"]["per_class"]) == {"1", "2", "3", "4", "5"}
    # Selection block keeps the candidates used for the comparison.
    assert body["selection"]["selected_classifier"] == "logreg"
    assert set(body["selection"]["candidates"]) == {"logreg", "linear_svc"}
    # Provenance and dependencies survive the round-trip.
    assert body["git_commit"] == "0" * 40
    assert body["git_dirty"] is False
    assert body["dependency_versions"]["python"] == platform.python_version()
    assert "created_at" in body


def test_model_info_returns_503_when_artifact_missing(tmp_path: Path) -> None:
    """The endpoint must refuse to lie about a model that is not loaded."""

    # Build a holder pointing at a real artifact (so ``load()`` would
    # succeed), then neutralize ``load()`` and clear the populated state.
    # ``lifespan`` calls ``holder.load()`` on startup, so we have to
    # prevent it from repopulating the fields we just zeroed.
    holder = ModelHolder(_artifact(tmp_path))
    holder.load = lambda: None  # type: ignore[method-assign]
    holder.pipeline = None
    holder.metadata = {}
    holder.label_names = {}
    holder.model_version = None
    with TestClient(create_app(holder=holder)) as test_client:
        response = test_client.get("/model-info")
    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "model_not_ready"
    assert body["request_id"]
    assert "could not be processed" in body["message"].lower()


# ---------------------------------------------------------------------------
# Model picker (``GET /models`` + ``POST /reload``)
# ---------------------------------------------------------------------------


VERSION_B = "20260823T120500Z-fedcba987654"


def _second_artifact(repo_root: Path, version: str = VERSION_B) -> Path:
    """Write a second immutable artifact under ``<repo_root>/models/<version>/``.

    The dashboard tests need at least two versions to exercise the picker.
    We build a brand new pipeline + manifest with the supplied version
    so ``validate_metadata`` accepts it without changes.
    """

    models_dir = repo_root / "models"
    return _artifact(models_dir, version=version)


# ``_artifact`` writes to ``<parent>/<version>/model.joblib``, so the
# caller passes the models dir and the desired version. Both fixtures
# above funnel through the same helper to keep checksum / manifest
# validation uniform.


def test_list_models_returns_newest_first_and_marks_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GET /models`` lists every version and tags the one in the holder."""

    monkeypatch.setattr("triage_ml.dev_api.app.REPO_ROOT", tmp_path)
    holder_path = _artifact(tmp_path / "models", version=VERSION)
    _second_artifact(tmp_path)
    # ``VERSION_B`` has a slightly later timestamp, so it must come first.
    assert VERSION_B > VERSION
    holder = ModelHolder(holder_path)
    with TestClient(create_app(holder=holder)) as test_client:
        with patch(
            "joblib.load", side_effect=AssertionError("must not deserialize during listing")
        ):
            response = test_client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["versions"] == [VERSION_B, VERSION]
    assert body["current"] == VERSION


def test_list_models_when_models_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GET /models`` returns an empty list (not 500) when ``models/`` is gone."""

    empty_root = tmp_path / "empty_repo"
    empty_root.mkdir()
    monkeypatch.setattr("triage_ml.dev_api.app.REPO_ROOT", empty_root)
    holder_path = _artifact(tmp_path / "models", version=VERSION)
    holder = ModelHolder(holder_path, registry_root=empty_root / "models")
    with TestClient(create_app(holder=holder)) as test_client:
        response = test_client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["versions"] == []
    # The holder still reports its loaded version even if the directory
    # disappeared afterwards — useful for surfacing state to the client.
    assert body["current"] == VERSION


def test_list_models_omits_incomplete_artifacts(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    holder_path = _artifact(models_dir, version=VERSION)
    invalid_dir = models_dir / VERSION_B
    invalid_dir.mkdir()
    (invalid_dir / "model.joblib").write_bytes(b"incomplete")
    holder = ModelHolder(holder_path)
    with TestClient(create_app(holder=holder)) as test_client:
        response = test_client.get("/models")
    assert response.json()["versions"] == [VERSION]


def test_unknown_route_uses_sanitized_error_contract(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error_code"] == "request_failed"
    assert response.headers["x-request-id"] == response.json()["request_id"]


def test_reload_swaps_holder_and_health_reflects_new_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``POST /reload`` validates the new version and updates the holder."""

    monkeypatch.setattr("triage_ml.dev_api.app.REPO_ROOT", tmp_path)
    holder_path = _artifact(tmp_path / "models", version=VERSION)
    _second_artifact(tmp_path)
    holder = ModelHolder(holder_path)
    with TestClient(create_app(holder=holder)) as test_client:
        before = test_client.get("/health").json()
        assert before["model_version"] == VERSION

        with patch("joblib.load", wraps=joblib.load) as mocked_load:
            response = test_client.post("/reload", json={"model_version": VERSION_B})
        assert mocked_load.call_count == 1

        assert response.status_code == 200
        assert response.json() == {
            "model_version": VERSION_B,
            "model_loaded": True,
        }
        after = test_client.get("/health").json()
        assert after["model_version"] == VERSION_B
        # ``/model-info`` must agree with the new holder state.
        info = test_client.get("/model-info").json()
        assert info["model_version"] == VERSION_B


def test_prediction_uses_one_holder_snapshot_during_reload(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    holder_path = _artifact(models_dir, version=VERSION)
    _artifact(models_dir, version=VERSION_B)
    holder = ModelHolder(holder_path)

    class ReloadingPipeline:
        def predict(self, _: list[str]) -> list[int]:
            holder.reload_to(VERSION_B)
            return [1]

    with TestClient(create_app(holder=holder)) as test_client:
        holder.pipeline = ReloadingPipeline()
        response = test_client.post(
            "/predict",
            json={
                "text": (
                    "We report a patient with an aggressive liver tumor requiring "
                    "surgical resection and histopathological evaluation."
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["model_version"] == VERSION
    assert holder.model_version == VERSION_B


def test_reload_returns_404_for_unknown_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown / malformed versions become ``404 model_not_found``."""

    monkeypatch.setattr("triage_ml.dev_api.app.REPO_ROOT", tmp_path)
    holder_path = _artifact(tmp_path / "models", version=VERSION)
    holder = ModelHolder(holder_path)
    with TestClient(create_app(holder=holder)) as test_client:
        response = test_client.post("/reload", json={"model_version": "not-a-real-version"})
    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "model_not_found"
    assert "not found" in body["message"].lower()
    assert body["request_id"]


def test_reload_rejects_when_request_body_is_empty(
    client: TestClient,
) -> None:
    """An empty body must trigger ``validation_failed`` (Pydantic)."""

    response = client.post("/reload", json={})
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation_failed"


def test_holder_reload_to_unknown_version_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ModelHolder.reload_to`` must refuse to mutate the holder on failure."""

    monkeypatch.setattr("triage_ml.dev_api.app.REPO_ROOT", tmp_path)
    holder_path = _artifact(tmp_path / "models", version=VERSION)
    holder = ModelHolder(holder_path)
    holder.load()
    original_version = holder.model_version
    with pytest.raises(FileNotFoundError):
        holder.reload_to("99999999T999999Z-deadbeef0000")
    # The holder keeps serving the previous model — never a half-loaded state.
    assert holder.model_version == original_version
    assert holder.pipeline is not None

"""Tests proving RBAC limits and sanitization inside the production API."""

import pytest
from fastapi.testclient import TestClient

from triage_ml.api.app import create_app
from triage_ml.api.settings import Settings, get_settings


class DummyPipeline:
    """Minimal deterministic pipeline used to isolate HTTP authorization tests."""

    classes_ = [1, 2, 3]

    def predict(self, texts: list[str]) -> list[int]:
        return [1]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.8, 0.1, 0.1]]


class DummyHolder:
    """Model holder test double that never touches the filesystem."""

    def __init__(self) -> None:
        self.pipeline = DummyPipeline()
        self.metadata = {"language": "en"}
        self.label_names = {1: "neoplasms", 2: "other", 3: "other"}
        self.model_version = "20260823T120000Z-0123456789ab"
        self.registry_root = "/tmp/models"

    def load(self) -> None:
        """The production lifespan calls load; this double is already ready."""

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def reload_to(self, version: str) -> str:
        self.model_version = version
        return version

    def snapshot(self) -> tuple[DummyPipeline, dict[str, str], dict[int, str], str]:
        return self.pipeline, self.metadata, self.label_names, self.model_version


def get_test_settings() -> Settings:
    return Settings(
        api_key_service="srv-" + "0" * 30,
        api_key_doctor="doc-" + "0" * 30,
        api_key_patient="pat-" + "0" * 30,
        log_level="INFO",
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a hermetic application with deterministic model and language checks."""

    monkeypatch.setattr("triage_ml.api.app.detect_language", lambda *args, **kwargs: None)
    app = create_app(holder=DummyHolder())
    app.dependency_overrides[get_settings] = get_test_settings

    with TestClient(app) as test_client:
        yield test_client


def test_patient_is_denied_prediction_and_leaks_nothing(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"text": "Patient comes in with severe chest pain and cardiovascular issues."},
        headers={"X-API-Key": "pat-" + "0" * 30},
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "clinician_review_required"
    assert "label" not in response.json()
    assert "score" not in response.json()
    assert "chest pain" not in response.text


def test_missing_credential_is_denied(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "A sufficiently long test message."})

    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthorized"


def test_doctor_can_predict_but_not_reload(client: TestClient) -> None:
    headers = {"X-API-Key": "doc-" + "0" * 30}

    prediction = client.post(
        "/predict",
        json={"text": "A patient has a documented cardiovascular condition."},
        headers=headers,
    )
    assert prediction.status_code == 200
    assert prediction.json()["label"] == 1

    reload_response = client.post(
        "/reload",
        json={"model_version": "20260823T120000Z-0123456789ab"},
        headers=headers,
    )
    assert reload_response.status_code == 403
    assert reload_response.json()["error_code"] == "forbidden"


def test_service_can_reload_but_cannot_predict(client: TestClient) -> None:
    headers = {"X-API-Key": "srv-" + "0" * 30}

    reload_response = client.post(
        "/reload",
        json={"model_version": "20260823T120000Z-0123456789ab"},
        headers=headers,
    )
    assert reload_response.status_code == 200

    prediction = client.post(
        "/predict",
        json={"text": "A patient has a documented cardiovascular condition."},
        headers=headers,
    )
    assert prediction.status_code == 403
    assert prediction.json()["error_code"] == "forbidden"

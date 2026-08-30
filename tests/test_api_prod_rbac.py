"""Tests proving RBAC limits and sanitization inside the production API."""

import pytest
from fastapi.testclient import TestClient

from triage_ml.api.app import create_app
from triage_ml.api.settings import Settings, get_settings
from triage_ml.dev_api.app import ModelHolder


# --- Mocks para isolar o teste do modelo real no disco ---

class DummyPipeline:
    classes_ = [1, 2, 3]

    def predict(self, texts: list[str]) -> list[int]:
        return [1]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.8, 0.1, 0.1]]


class DummyHolder(ModelHolder):
    """Emula o ModelHolder sem tocar o filesystem ou necessitar de artefatos reais."""

    def __init__(self) -> None:
        self.loaded = True
        self.pipeline = DummyPipeline()
        self.metadata = {"language": "en"}
        self.label_names = {1: "neoplasms", 2: "other", 3: "other"}
        self.model_version = "20260823T120000Z-dummy"
        self.registry_root = "/tmp/models"

    def snapshot(self) -> tuple:
        return self.pipeline, self.metadata, self.label_names, self.model_version


# --- Configuração Explícita ---

def get_test_settings() -> Settings:
    return Settings(
        api_key_service="srv-" + "0" * 30,
        api_key_doctor="doc-" + "0" * 30,
        api_key_patient="pat-" + "0" * 30,
        log_level="INFO",
    )


@pytest.fixture
def client() -> TestClient:
    app = create_app(holder=DummyHolder())
    # Injeta explicitamente as configurações sem alterar env vars globais (os.environ)
    app.dependency_overrides[get_settings] = get_test_settings
    
    with TestClient(app) as test_client:
        yield test_client


# --- Testes de RBAC e Sanitização ---

def test_patient_is_denied_prediction_and_leaks_nothing(client: TestClient) -> None:
    # O texto é longo o suficiente em inglês para não falhar na checagem de idioma antes do RBAC
    response = client.post(
        "/predict",
        json={"text": "Patient comes in with severe chest pain and cardiovascular issues."},
        headers={"X-API-Key": "pat-" + "0" * 30}
    )
    assert response.status_code == 403
    
    body = response.json()
    assert body["error_code"] == "clinician_review_required"
    
    # Valida ausência total de vazamento
    assert "label" not in body
    assert "score" not in body
    assert "chest pain" not in response.text


def test_missing_or_invalid_credential_fails_safe(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "Dummy text for validation."})
    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthorized"


def test_doctor_can_predict_but_not_reload(client: TestClient) -> None:
    # Predição bem-sucedida (Status 200 determinístico graças ao mock)
    predict_res = client.post(
        "/predict",
        json={"text": "We report a patient with an aggressive liver tumor requiring surgery."},
        headers={"X-API-Key": "doc-" + "0" * 30}
    )
    assert predict_res.status_code == 200
    assert predict_res.json()["label"] == 1

    # Reload bloqueado (Apenas service pode recarregar)
    reload_res = client.post(
        "/reload",
        json={"model_version": "some-version"},
        headers={"X-API-Key": "doc-" + "0" * 30}
    )
    assert reload_res.status_code == 403
    assert reload_res.json()["error_code"] == "forbidden"


def test_service_is_denied_prediction(client: TestClient) -> None:
    # O service gerencia ciclos do modelo, mas não consome predições (restrito a doctor)
    response = client.post(
        "/predict",
        json={"text": "Cardiovascular pain requiring surgical intervention."},
        headers={"X-API-Key": "srv-" + "0" * 30}
    )
    assert response.status_code == 403
    assert response.json()["error_code"] == "forbidden"
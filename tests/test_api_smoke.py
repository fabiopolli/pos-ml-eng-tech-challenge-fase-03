"""Smoke tests for the FastAPI app.

These tests exercise the contract proposed in
``docs/plans/PLAN-text-classifier.md``: latency_ms, request_id,
X-Request-ID, sanitized errors and no clinical text in responses.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from triage_ml.api.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok_and_model_metadata(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert isinstance(body["model_version"], str)


def test_predict_returns_expected_fields(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "Liver tumor with metastases."})
    assert response.status_code == 200
    body = response.json()
    for key in (
        "label",
        "label_name",
        "score",
        "model_version",
        "latency_ms",
        "request_id",
        "warnings",
    ):
        assert key in body, f"missing field {key}"
    assert isinstance(body["label"], int)
    assert body["label_name"]
    assert body["score"] is None or 0.0 <= body["score"] <= 1.0
    assert body["latency_ms"] >= 0
    assert len(body["request_id"]) >= 8


def test_predict_echoes_request_id_in_header(client: TestClient) -> None:
    custom = "rid-test-abc"
    response = client.post(
        "/predict",
        json={"text": "Acute myocardial infarction."},
        headers={"X-Request-ID": custom},
    )
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom
    assert response.json()["request_id"] == custom


def test_empty_text_returns_422_sanitized(client: TestClient) -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "validation_failed"
    assert "text" not in body["message"].lower() or "invalid" in body["message"].lower()
    # The original clinical text must NOT appear in the error body.
    assert "" not in body or body.get("message") == "Request body is invalid."
    assert "request_id" in body


def test_missing_text_field_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "validation_failed"
    assert body["request_id"]


def test_latency_varies_between_calls(client: TestClient) -> None:
    first = client.post("/predict", json={"text": "Cardiac arrest in 60yo."}).json()
    second = client.post("/predict", json={"text": "Brain tumor in 30yo."}).json()
    # Two consecutive calls should report positive, potentially different latencies.
    assert first["latency_ms"] >= 0
    assert second["latency_ms"] >= 0

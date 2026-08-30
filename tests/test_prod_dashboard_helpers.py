"""Hermetic tests for the production Streamlit dashboard helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "front" / "app_prod.py"


@pytest.fixture
def dashboard_module():
    """Import dashboard helpers without starting a Streamlit server."""

    spec = importlib.util.spec_from_file_location("app_prod", DASHBOARD_PATH)
    assert spec is not None and spec.loader is not None, "could not load app_prod"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - streamlit missing
        pytest.skip(f"streamlit not installed: {exc}")
    return module


class FakeResponse:
    """Minimal ``requests.Response`` substitute for HTTP helper tests."""

    def __init__(self, *, status_code: int, body: object, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body, ensure_ascii=False)

    def json(self) -> object:
        return self._body


def dashboard_environment() -> dict[str, str]:
    """Return synthetic, non-secret configuration for a dashboard session."""

    return {
        "TRIAGE_ML_PROD_API_URL": "http://127.0.0.1:8000",
        "TRIAGE_ML_API_KEY_DOCTOR": "doc-" + "0" * 30,
        "TRIAGE_ML_DASHBOARD_DOCTOR_USERNAME": "doctor-demo",
        "TRIAGE_ML_DASHBOARD_DOCTOR_PASSWORD": "doctor-password",
        "TRIAGE_ML_DASHBOARD_PATIENT_USERNAME": "patient-demo",
        "TRIAGE_ML_DASHBOARD_PATIENT_PASSWORD": "patient-password",
    }


def test_config_requires_all_runtime_secrets(dashboard_module):
    environment = dashboard_environment()
    del environment["TRIAGE_ML_API_KEY_DOCTOR"]

    with pytest.raises(RuntimeError, match="TRIAGE_ML_API_KEY_DOCTOR"):
        dashboard_module.load_config(environment)


def test_config_normalizes_the_api_url(dashboard_module):
    environment = dashboard_environment()
    environment["TRIAGE_ML_PROD_API_URL"] = "http://127.0.0.1:8000/"

    config = dashboard_module.load_config(environment)

    assert config.api_url == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    ("username", "password", "expected_role"),
    [
        ("doctor-demo", "doctor-password", "doctor"),
        ("patient-demo", "patient-password", "patient"),
        ("doctor-demo", "wrong-password", None),
        ("unknown", "patient-password", None),
    ],
)
def test_login_maps_only_valid_credentials_to_roles(
    dashboard_module, username: str, password: str, expected_role: str | None
):
    config = dashboard_module.load_config(dashboard_environment())

    assert dashboard_module.authenticate(username, password, config) == expected_role


def test_only_doctor_role_can_request_prediction(dashboard_module):
    assert dashboard_module.can_request_prediction("doctor") is True
    assert dashboard_module.can_request_prediction("patient") is False
    assert dashboard_module.can_request_prediction("service") is False


def test_doctor_prediction_uses_server_side_doctor_key(dashboard_module):
    config = dashboard_module.load_config(dashboard_environment())
    response = FakeResponse(
        status_code=200,
        body={"label": 1, "label_name": "neoplasms"},
        headers={"X-Request-ID": "request-123"},
    )

    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        result = dashboard_module.doctor_predict(config, "Synthetic English clinical text.")

    assert result.status_code == 200
    assert mocked.call_args.kwargs["url"] == "http://127.0.0.1:8000/predict"
    assert mocked.call_args.kwargs["headers"]["X-API-Key"] == config.doctor_api_key
    assert mocked.call_args.kwargs["json"] == {"text": "Synthetic English clinical text."}
    assert mocked.call_args.kwargs["allow_redirects"] is False


def test_health_request_never_sends_doctor_key(dashboard_module):
    config = dashboard_module.load_config(dashboard_environment())
    response = FakeResponse(status_code=200, body={"status": "ok", "model_loaded": True})

    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        dashboard_module.check_health(config)

    assert mocked.call_args.kwargs["headers"] == {"Accept": "application/json"}


def test_api_key_is_not_present_in_response_representation(dashboard_module):
    config = dashboard_module.load_config(dashboard_environment())
    response = FakeResponse(status_code=200, body={"status": "ok"})

    with patch.object(dashboard_module.requests, "request", return_value=response):
        result = dashboard_module.check_health(config)

    assert config.doctor_api_key not in repr(result)

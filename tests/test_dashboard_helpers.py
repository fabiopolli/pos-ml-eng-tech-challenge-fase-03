"""Hermetic tests for the smoke dashboard helpers.

The dashboard itself runs inside Streamlit, which is heavy to spin up in
unit tests. We cover the deterministic helpers (``_check_health``,
``_post_predict``, ``_render_response`` and the preset table) by mocking
``requests.request`` so we don't need a live API.

Streamlit is imported lazily inside the module (see ``front/app_smoke.py``
import guard below) — when not available, only the pure helpers execute.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "front" / "app_smoke.py"


@pytest.fixture
def dashboard_module():
    """Import the dashboard module bypassing the Streamlit runtime."""

    spec = importlib.util.spec_from_file_location("app_smoke", DASHBOARD_PATH)
    assert spec is not None and spec.loader is not None, "could not load app_smoke"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass decorators find the module name.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - streamlit missing
        pytest.skip(f"streamlit not installed: {exc}")
    return module


class _FakeResponse:
    """Mimics ``requests.Response`` enough for the dashboard helpers."""

    def __init__(
        self,
        *,
        status_code: int,
        body: object,
        headers: dict[str, str] | None = None,
        elapsed_seconds: float = 0.123,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.elapsed = type("elapsed", (), {"total_seconds": lambda self: elapsed_seconds})()
        self.text = json_dumps(body)

    def json(self) -> object:
        return self._body


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def test_health_calls_endpoint_and_parses_response(dashboard_module):
    response = _FakeResponse(
        status_code=200,
        body={
            "status": "ok",
            "model_version": "20260823T135811Z-bed2194376bc",
            "model_loaded": True,
        },
        headers={
            "X-Request-ID": "abc123",
            "Server-Timing": "detect;dur=0.500, predict;dur=1.234",
        },
    )
    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        result = dashboard_module._check_health("http://api.example.com")
    assert mocked.call_count == 1
    method = mocked.call_args.kwargs.get("method") or mocked.call_args.args[0]
    url = mocked.call_args.kwargs.get("url") or mocked.call_args.args[1]
    assert method == "GET"
    assert url == "http://api.example.com/health"
    assert mocked.call_args.kwargs["timeout"] == dashboard_module.REQUEST_TIMEOUT_SECONDS
    assert result.status_code == 200
    assert result.body["model_loaded"] is True
    assert result.server_timing == "detect;dur=0.500, predict;dur=1.234"
    assert result.request_id == "abc123"
    assert result.elapsed_ms == pytest.approx(123.0, abs=0.01)


def test_predict_posts_json_payload(dashboard_module):
    response = _FakeResponse(
        status_code=200,
        body={
            "label": 1,
            "label_name": "neoplasms",
            "score": None,
            "model_version": "20260823T135811Z-bed2194376bc",
            "latency_ms": 1.5,
            "request_id": "deadbeef",
            "warnings": [],
        },
    )
    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        result = dashboard_module._post_predict("http://api.example.com/", "Tumor growth")
    method = mocked.call_args.kwargs.get("method") or mocked.call_args.args[0]
    url = mocked.call_args.kwargs.get("url") or mocked.call_args.args[1]
    assert method == "POST"
    assert url == "http://api.example.com/predict"
    assert mocked.call_args.kwargs["json"] == {"text": "Tumor growth"}
    assert result.status_code == 200
    assert result.body["label_name"] == "neoplasms"


def test_invalid_json_body_is_wrapped_in_raw(dashboard_module):
    response = _FakeResponse(status_code=500, body={})
    response.json = MagicMock(side_effect=ValueError("not json"))
    response.text = "<html>oops</html>"
    with patch.object(dashboard_module.requests, "request", return_value=response):
        result = dashboard_module._post_predict("http://api.example.com", "anything")
    assert result.status_code == 500
    assert result.body == {"raw": "<html>oops</html>"}


def test_request_exception_surfaces_to_caller(dashboard_module):
    with patch.object(
        dashboard_module.requests,
        "request",
        side_effect=dashboard_module.requests.RequestException("boom"),
    ):
        with pytest.raises(dashboard_module.requests.RequestException):
            dashboard_module._check_health("http://api.example.com")


def test_language_presets_have_expected_error_codes(dashboard_module):
    presets = dashboard_module.LANGUAGE_PRESETS
    assert presets["Texto curto (<20 chars)"]["expected_error_code"] == (
        "text_too_short_for_language_check"
    )
    assert presets["Confiança baixa (mock)"]["expected_error_code"] == ("indeterminate_language")
    assert presets["Idioma fora do allow-list"]["expected_error_code"] == ("unsupported_language")
    assert presets["Inglês válido"]["expected_error_code"] is None
    # Texts must satisfy the minimum length except for the short preset.
    for name, payload in presets.items():
        if name == "Texto curto (<20 chars)":
            assert len(payload["text"]) < 20
        else:
            assert len(payload["text"]) >= 20


def test_url_stripping_handles_trailing_slash(dashboard_module):
    response = _FakeResponse(status_code=200, body={"status": "ok", "model_loaded": True})
    with patch.object(dashboard_module.requests, "request", return_value=response) as mocked:
        dashboard_module._check_health("http://api.example.com///")
    method = mocked.call_args.kwargs.get("method") or mocked.call_args.args[0]
    url = mocked.call_args.kwargs.get("url") or mocked.call_args.args[1]
    assert method == "GET"
    assert url == "http://api.example.com/health"

"""Security-focused tests for production API settings, rate limits, and logs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from triage_ml.api.app import create_app
from triage_ml.api.ratelimit import get_api_key_fingerprint
from triage_ml.api.settings import Settings, get_settings

SERVICE_KEY = "srv-" + "0" * 30
DOCTOR_KEY = "doc-" + "0" * 30
PATIENT_KEY = "pat-" + "0" * 30


class DummyPipeline:
    """Deterministic pipeline used without loading a model artifact."""

    classes_ = [1, 2, 3]

    def predict(self, texts: list[str]) -> list[int]:
        return [1]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.8, 0.1, 0.1]]


class FailingPipeline(DummyPipeline):
    """Pipeline that fails without retaining a clinical text in its error."""

    def predict(self, texts: list[str]) -> list[int]:
        raise RuntimeError("predict failed")


class DummyHolder:
    """Model-holder double that never reads or writes the filesystem."""

    def __init__(self, pipeline: DummyPipeline | None = None) -> None:
        self.pipeline = pipeline if pipeline is not None else DummyPipeline()
        self.metadata = {"language": "en"}
        self.label_names = {1: "neoplasms", 2: "other", 3: "other"}
        self.model_version = "20260823T120000Z-0123456789ab"
        self.registry_root = "."

    def load(self) -> None:
        """The production lifespan calls load; this double is already ready."""

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def reload_to(self, version: str) -> str:
        self.model_version = version
        return version

    def snapshot(self) -> tuple[DummyPipeline | None, dict[str, str], dict[int, str], str]:
        return self.pipeline, self.metadata, self.label_names, self.model_version


def make_settings(**overrides: str) -> Settings:
    """Provide complete test settings while allowing a focused override."""

    values = {
        "api_key_service": SERVICE_KEY,
        "api_key_doctor": DOCTOR_KEY,
        "api_key_patient": PATIENT_KEY,
    }
    values.update(overrides)
    return Settings(**values)


@contextmanager
def api_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    holder: DummyHolder | None = None,
    settings: Settings | None = None,
):
    """Build a hermetic app instance with a deterministic language check."""

    monkeypatch.setattr("triage_ml.api.app.detect_language", lambda *args, **kwargs: None)
    active_settings = settings or make_settings()
    app = create_app(holder=holder or DummyHolder(), settings=active_settings)
    app.dependency_overrides[get_settings] = lambda: active_settings
    with TestClient(app) as client:
        yield client


def request_with_api_key(api_key: str | None) -> Request:
    """Create the minimal ASGI request used by the limiter key function."""

    headers = [] if api_key is None else [(b"x-api-key", api_key.encode("utf-8"))]
    return Request({"type": "http", "method": "POST", "path": "/predict", "headers": headers})


def test_settings_reject_unknown_values() -> None:
    with pytest.raises(ValidationError):
        make_settings(unexpected_option="not allowed")


def test_settings_accepts_explicit_valid_values() -> None:
    settings = make_settings(ratelimit_predict="2/minute")

    assert settings.ratelimit_predict == "2/minute"


def test_settings_does_not_read_unrelated_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "must-not-enter-settings-validation"
    (tmp_path / ".env").write_text(f"DAGSHUB_USER_TOKEN={sentinel}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = make_settings()

    assert sentinel not in repr(settings)


@pytest.mark.parametrize(
    "field",
    ["api_key_service", "api_key_doctor", "api_key_patient"],
)
def test_settings_reject_short_api_keys(field: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: "too-short"})


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        (None, "anonymous"),
        (DOCTOR_KEY, "e8b26f0feedcbc8d4a5b3600409a73ba38f144c8dbdc76df9d11e03663d2beb6"),
        (PATIENT_KEY, "59f899a07088b886354246a27fd32abc3cb1b21b2f4609bb1c50b68f5bf04732"),
    ],
)
def test_rate_limit_identifier_never_returns_api_key(api_key: str | None, expected: str) -> None:
    fingerprint = get_api_key_fingerprint(request_with_api_key(api_key))

    assert fingerprint == expected
    assert api_key is None or api_key not in fingerprint


def test_predict_is_limited_per_request_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(ratelimit_predict="2/minute")
    headers = {"X-API-Key": DOCTOR_KEY}

    with api_client(monkeypatch, settings=settings) as client:
        for _ in range(2):
            response = client.post(
                "/predict", json={"text": "A long enough English test message."}, headers=headers
            )
            assert response.status_code == 200

        limited = client.post(
            "/predict", json={"text": "A long enough English test message."}, headers=headers
        )

    assert limited.status_code == 429


def test_reload_is_limited_per_request_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(ratelimit_default="1/minute")
    headers = {"X-API-Key": SERVICE_KEY}
    body = {"model_version": "20260823T120000Z-0123456789ab"}

    with api_client(monkeypatch, settings=settings) as client:
        assert client.post("/reload", json=body, headers=headers).status_code == 200
        limited = client.post("/reload", json=body, headers=headers)

    assert limited.status_code == 429


def test_patient_text_is_absent_from_structured_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class CapturingLogger:
        def info(self, *args: object, **kwargs: object) -> None:
            entries.append(("info", args, kwargs))

        def error(self, *args: object, **kwargs: object) -> None:
            entries.append(("error", args, kwargs))

    monkeypatch.setattr("triage_ml.api.app.logger", CapturingLogger())
    clinical_text = "sentinel clinical phrase should never enter logs"

    with api_client(monkeypatch) as client:
        response = client.post(
            "/predict", json={"text": clinical_text}, headers={"X-API-Key": PATIENT_KEY}
        )

    assert response.status_code == 403
    assert clinical_text not in response.text
    assert clinical_text not in repr(entries)


def test_prediction_failure_does_not_log_or_return_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class CapturingLogger:
        def info(self, *args: object, **kwargs: object) -> None:
            entries.append(("info", args, kwargs))

        def error(self, *args: object, **kwargs: object) -> None:
            entries.append(("error", args, kwargs))

    monkeypatch.setattr("triage_ml.api.app.logger", CapturingLogger())
    clinical_text = "private tumor description must not be disclosed"

    with api_client(monkeypatch, holder=DummyHolder(FailingPipeline())) as client:
        response = client.post(
            "/predict", json={"text": clinical_text}, headers={"X-API-Key": DOCTOR_KEY}
        )

    assert response.status_code == 500
    assert clinical_text not in response.text
    assert clinical_text not in repr(entries)


def test_health_reports_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    with api_client(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info_returns_sanitized_not_ready_error(monkeypatch: pytest.MonkeyPatch) -> None:
    unloaded_holder = DummyHolder()
    unloaded_holder.pipeline = None

    with api_client(monkeypatch, holder=unloaded_holder) as client:
        response = client.get("/model-info")

    assert response.status_code == 503
    assert response.json()["error_code"] == "model_not_ready"

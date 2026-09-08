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

    def reload_to(self, version: str, *, variant: str = "sklearn") -> str:
        self.model_version = version
        return version

    def snapshot(self) -> tuple[DummyPipeline | None, dict[str, str], dict[int, str], str]:
        return self.pipeline, self.metadata, self.label_names, self.model_version

    def prediction_snapshot(
        self, variant: str
    ) -> tuple[DummyPipeline | None, dict[str, str], dict[int, str], str]:
        return self.snapshot()

    def ensure_variant_ready(self, variant: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("model is not loaded")


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


def test_settings_reject_duplicate_api_keys() -> None:
    """Reusing the same value across roles would silently collapse role checks."""

    with pytest.raises(ValidationError, match="must be distinct"):
        make_settings(api_key_service=DOCTOR_KEY, api_key_doctor=DOCTOR_KEY)


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


@pytest.mark.parametrize("api_key", [None, DOCTOR_KEY, PATIENT_KEY])
def test_rate_limit_identifier_never_returns_api_key(api_key: str | None) -> None:
    fingerprint = get_api_key_fingerprint(request_with_api_key(api_key))

    if api_key is None:
        assert fingerprint == "anonymous"
    else:
        # The fingerprint is salted HMAC-SHA-256 prefixed with "k:" so an
        # attacker rotating X-API-Key headers cannot pre-compute buckets or
        # cross-reference with rainbow tables. We assert the structural
        # invariants here; the salt keeps the value opaque from the test.
        assert fingerprint.startswith("k:")
        assert len(fingerprint) == 2 + 64
        assert api_key not in fingerprint


def test_rate_limit_identifier_is_stable_for_same_key() -> None:
    """A given API key must always resolve to the same fingerprint within a
    process. Otherwise the rate limit bucket would reset on every call."""

    fingerprint_a = get_api_key_fingerprint(request_with_api_key(DOCTOR_KEY))
    fingerprint_b = get_api_key_fingerprint(request_with_api_key(DOCTOR_KEY))

    assert fingerprint_a == fingerprint_b


def test_rate_limit_identifier_changes_when_key_changes() -> None:
    """Different API keys must produce different fingerprints, otherwise a
    caller rotating the header could share a bucket with another caller."""

    fingerprint_a = get_api_key_fingerprint(request_with_api_key(DOCTOR_KEY))
    fingerprint_b = get_api_key_fingerprint(request_with_api_key(PATIENT_KEY))

    assert fingerprint_a != fingerprint_b


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


def test_health_returns_503_when_model_is_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifespan did not load the model: /health must surface 503."""

    monkeypatch.setattr("triage_ml.api.app.detect_language", lambda *args, **kwargs: None)
    settings = make_settings()
    holder = DummyHolder.__new__(DummyHolder)
    holder.pipeline = None
    holder.metadata = {"language": "en"}
    holder.label_names = {}
    holder.model_version = None
    holder.registry_root = "."
    holder.load = lambda: None  # type: ignore[method-assign]
    app = create_app(holder=holder, settings=settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_prod_lifespan_blocks_when_languages_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prod deployment with mismatched language config must abort startup."""

    from triage_ml.dev_api import config as dev_api_config

    class PtHolder(DummyHolder):
        @property
        def loaded(self) -> bool:
            return True

    holder = PtHolder()
    holder.metadata = {"language": "pt"}
    dev_api_config.reset_api_config_cache()
    monkeypatch.setattr(
        dev_api_config,
        "load_api_config",
        lambda path=None: dev_api_config.ApiConfig(supported_languages=frozenset({"en"})),
    )

    try:
        with pytest.raises(RuntimeError, match="supported languages"):
            with TestClient(create_app(holder=holder, settings=make_settings())):
                pass
    finally:
        dev_api_config.reset_api_config_cache()


def test_model_info_requires_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    with api_client(monkeypatch) as client:
        response = client.get("/model-info")

    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthorized"


def test_models_requires_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    with api_client(monkeypatch) as client:
        response = client.get("/models")

    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthorized"


def test_model_info_returns_sanitized_not_ready_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even after auth, /model-info must surface a sanitized 503 when no
    artifact is loaded (no leakage of internal pipeline state)."""

    unloaded_holder = DummyHolder()
    unloaded_holder.pipeline = None

    with api_client(monkeypatch, holder=unloaded_holder) as client:
        response = client.get("/model-info", headers={"X-API-Key": DOCTOR_KEY})

    assert response.status_code == 503
    assert response.json()["error_code"] == "model_not_ready"

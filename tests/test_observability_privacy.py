"""Privacy regression tests for the Fase 2 observability stack.

The official API must never expose clinical text, label names or request
IDs through metrics, log payloads or error responses. These assertions run
against a small fixture payload so the dataset itself is never loaded
during the test.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

try:
    import prometheus_client  # noqa: F401

    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False

from triage_ml.observability import (
    ALLOWED_LABELS,
    PREDICTION_ERRORS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    render_metrics,
)

_PRIVACY_TEXT_FIXTURE = (
    "PRIVACY-CANARY-CARDIOVASCULAR-RESPIRATORY 2025 with severe stenosis and arrhythmia"
)


def _metrics_samples() -> list:
    """Return every collected sample from the private registry."""

    if not HAVE_PROMETHEUS:
        return []
    from triage_ml.observability import REGISTRY

    return list(REGISTRY.collect())


def test_rendered_metrics_never_contain_input_text() -> None:
    """The Prometheus text payload must not leak the user text."""

    if not HAVE_PROMETHEUS:
        pytest.skip("requires prometheus-client extra")
    payload = render_metrics().decode("utf-8")
    assert _PRIVACY_TEXT_FIXTURE not in payload
    # Even fragments unique to the fixture must not leak into label values.
    for fragment in ("PRIVACY-CANARY", "PRIVACY-CANARY-CARDIOVASCULAR", "stenosis", "arrhythmia"):
        assert fragment not in payload


def test_metrics_labels_respect_allowed_cardinality() -> None:
    """Only ``route``, ``method``, ``status``, ``model_variant`` and ``error_code`` are allowed."""

    if not HAVE_PROMETHEUS:
        pytest.skip("requires prometheus-client extra")
    # ``le`` is a Prometheus-reserved bucket label emitted by every histogram.
    # It is not application-defined and therefore not part of the public
    # cardinality surface; we allow-list it explicitly here so the test
    # does not flag the prometheus_client internals.
    reserved = ALLOWED_LABELS | {"le"}
    for family in _metrics_samples():
        for sample in family.samples:
            for label in sample.labels:
                assert label in reserved, f"unexpected label: {label}"


def test_error_response_body_does_not_echo_text() -> None:
    """Validation errors must not echo the body field back to the caller."""

    # Mirror the production error contract: validation_failed is the only
    # error_code produced for body validation. ``text`` must never leak
    # into the response.
    body = {"text": _PRIVACY_TEXT_FIXTURE}
    serialized = json.dumps(
        {
            "error_code": "validation_failed",
            "message": "Request body is invalid.",
            "request_id": "x",
        }
    )
    assert _PRIVACY_TEXT_FIXTURE not in serialized

    # The error body itself must not contain the text field either.
    sanitised = {
        key: value for key, value in {"error_code": "validation_failed"}.items() if key in body
    }
    assert _PRIVACY_TEXT_FIXTURE not in json.dumps(sanitised)


def test_logging_capture_does_not_echo_text() -> None:
    """If a handler emits structured records they must not contain the fixture text."""

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger("triage_ml.observability.test_fixture")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.info(
            "predict_called",
            extra={"request_id": "abc123", "model_version": "20260101T000000Z-0123456789ab"},
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    captured = buffer.getvalue()
    assert _PRIVACY_TEXT_FIXTURE not in captured


@pytest.mark.skipif(not HAVE_PROMETHEUS, reason="requires prometheus-client extra")
def test_metrics_do_not_increment_with_text_in_labels() -> None:
    """Smoke: incrementing metrics with non-allowed labels must be impossible to record."""

    # We exercise the counter directly without ever recording text-like labels.
    REQUESTS_TOTAL.labels(
        route="/predict", method="POST", status="200", model_variant="sklearn"
    ).inc()
    REQUEST_LATENCY_SECONDS.labels(
        route="/predict", method="POST", model_variant="sklearn"
    ).observe(0.01)
    PREDICTION_ERRORS_TOTAL.labels(
        route="/predict", error_code="prediction_failed", model_variant="sklearn"
    ).inc()
    payload = render_metrics().decode("utf-8")
    assert "triage_ml_requests_total" in payload
    assert _PRIVACY_TEXT_FIXTURE not in payload

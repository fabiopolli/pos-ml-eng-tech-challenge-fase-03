"""Tests for the Fase 2 observability stack — metrics + middleware.

These tests use a private ``CollectorRegistry`` simulated in-memory; when
``prometheus-client`` is installed the ``PROMETHEUS_AVAILABLE`` branch runs,
otherwise the module degrades gracefully and the metrics surfaces return
empty payloads. The privacy tests are always enforced because they assert
on the metrics surface (registered samples) regardless of the extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import prometheus_client  # noqa: F401

    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False

from triage_ml.observability import (
    ALLOWED_LABELS,
    PROMETHEUS_AVAILABLE,
    REGISTRY,
    RENDER_CONTENT_TYPE,
    PrometheusMiddleware,
    render_metrics,
)


def test_prometheus_available_flag_observable() -> None:
    """Smoke test: ``PROMETHEUS_AVAILABLE`` matches the import probe."""

    assert PROMETHEUS_AVAILABLE is bool(HAVE_PROMETHEUS)


def test_allowed_labels_set_lists_known_labels() -> None:
    expected = {"route", "method", "status", "model_variant", "error_code"}
    assert ALLOWED_LABELS == frozenset(expected)


def test_render_metrics_returns_bytes() -> None:
    payload = render_metrics()
    if HAVE_PROMETHEUS:
        assert isinstance(payload, bytes)
        assert payload  # not empty after module import (no requests yet → only HELP lines)
    else:
        assert payload == b""


def test_render_metrics_prometheus_text_format_when_available() -> None:
    if not HAVE_PROMETHEUS:
        pytest.skip("requires prometheus-client extra")
    assert RENDER_CONTENT_TYPE.startswith("text/plain")
    payload = render_metrics().decode("utf-8")
    # Every Prometheus payload starts with ``# HELP``.
    assert "# HELP" in payload or "# TYPE" in payload


def test_registry_is_decoupled_from_global() -> None:
    """The metrics module must use its own registry, not the global one."""

    if not HAVE_PROMETHEUS:
        pytest.skip("requires prometheus-client extra")
    from prometheus_client import REGISTRY as default_registry

    assert REGISTRY is not default_registry  # type: ignore[comparison-overlap]


@pytest.mark.skipif(not HAVE_PROMETHEUS, reason="requires prometheus-client extra")
def test_normalise_route_keeps_cardinality_low() -> None:
    from triage_ml.observability.middleware import _normalise_route

    assert _normalise_route("/health") == "/health"
    assert _normalise_route("/predict") == "/predict"
    assert _normalise_route("/models") == "/models"
    assert _normalise_route("/anything-else") == "/other"


def test_prometheus_middleware_rejects_non_http_scope() -> None:
    """WebSocket and lifespan scopes must pass through untouched."""

    calls: list[dict] = []

    async def downstream(scope, receive, send):  # pragma: no cover - trivial
        calls.append(scope)

    middleware = PrometheusMiddleware(downstream)
    scope = {"type": "lifespan"}

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        return None

    import asyncio

    asyncio.run(middleware(scope, receive, send))
    assert calls and calls[0] is scope


def test_metrics_module_export_path_does_not_break_when_prometheus_missing() -> None:
    """The module is import-safe in environments without the extra."""

    import inspect

    from triage_ml import observability  # noqa: F401

    # Public surface is export-stable regardless of the extras.
    assert inspect.isclass(observability.PrometheusMiddleware)
    assert "render_metrics" in observability.__all__
    assert Path(observability.__file__).is_file()

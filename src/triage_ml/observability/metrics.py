"""Prometheus metrics primitives for the official API (Fase 2, Etapa 6).

The module exposes a private ``CollectorRegistry`` to avoid leaking metrics
from third-party libraries via the default registry. When
``prometheus-client`` is not installed, the metrics functions degrade
gracefully — ``render_metrics`` returns an empty payload and the middleware
returns without observing, so the production API never crashes on a missing
extras package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:  # pragma: no cover - exercised through the optional group
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through the optional group
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry as _CollectorRegistry
else:
    _CollectorRegistry = object


# Keep the actual registry resolvable at runtime so the FastAPI app can
# render the same payload via ``render_metrics()``.
REGISTRY: _CollectorRegistry
REQUESTS_TOTAL: object
REQUEST_LATENCY_SECONDS: object
PREDICTION_ERRORS_TOTAL: object


def _build_registry() -> None:
    global REGISTRY, REQUESTS_TOTAL, REQUEST_LATENCY_SECONDS, PREDICTION_ERRORS_TOTAL
    if PROMETHEUS_AVAILABLE:
        REGISTRY = CollectorRegistry()
        REQUESTS_TOTAL = Counter(
            "triage_ml_requests_total",
            "Total API requests processed by the official Triage ML service.",
            labelnames=("route", "method", "status", "model_variant"),
            registry=REGISTRY,
        )
        REQUEST_LATENCY_SECONDS = Histogram(
            "triage_ml_request_latency_seconds",
            "Latency of API requests (route, method, model_variant).",
            labelnames=("route", "method", "model_variant"),
            registry=REGISTRY,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        PREDICTION_ERRORS_TOTAL = Counter(
            "triage_ml_prediction_errors_total",
            "Errors emitted by the /predict endpoint, labelled with error_code.",
            labelnames=("route", "error_code", "model_variant"),
            registry=REGISTRY,
        )
    else:  # pragma: no cover - no-op stand-ins
        REGISTRY = None  # type: ignore[assignment]
        REQUESTS_TOTAL = REQUEST_LATENCY_SECONDS = PREDICTION_ERRORS_TOTAL = None


_build_registry()

# Cardinality guard: only the labels enumerated here may appear on metrics.
ALLOWED_LABELS = frozenset({"route", "method", "status", "model_variant", "error_code"})
FORBIDDEN_LABEL_HINTS = ("text", "label_name", "request_id")

RENDER_CONTENT_TYPE = CONTENT_TYPE_LATEST


def render_metrics() -> bytes:
    """Return the Prometheus text payload for the official API."""

    if not PROMETHEUS_AVAILABLE:
        return b""
    from prometheus_client import generate_latest

    return generate_latest(REGISTRY)

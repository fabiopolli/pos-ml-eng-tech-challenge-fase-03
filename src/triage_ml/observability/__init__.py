"""Observability primitives for the official API (Fase 2, Etapa 6).

The module keeps all Prometheus imports behind optional try-blocks so the
production image stays slim. ``prometheus-client`` is the only required
extras for the production API image once Etapa 6 is in scope; the
optimization DAG does not depend on observability.

Cardinality discipline:

* The only allowed labels are ``route``, ``method``, ``status`` and
  ``model_variant``.
* ``text``, ``label_name`` and ``request_id`` are explicitly forbidden in
  metrics labels (see ``tests/test_observability_privacy.py``).
* Histogram buckets are fixed at module-level so the same quantiles are
  comparable across variants and slices.
"""

from triage_ml.observability.metrics import (
    ALLOWED_LABELS,
    PREDICTION_ERRORS_TOTAL,
    PROMETHEUS_AVAILABLE,
    REGISTRY,
    RENDER_CONTENT_TYPE,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    render_metrics,
)
from triage_ml.observability.middleware import PrometheusMiddleware

__all__ = [
    "ALLOWED_LABELS",
    "PREDICTION_ERRORS_TOTAL",
    "PROMETHEUS_AVAILABLE",
    "PrometheusMiddleware",
    "REGISTRY",
    "RENDER_CONTENT_TYPE",
    "REQUEST_LATENCY_SECONDS",
    "REQUESTS_TOTAL",
    "render_metrics",
]

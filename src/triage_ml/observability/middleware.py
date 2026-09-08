"""HTTP middleware for Prometheus metrics (Fase 2, Etapa 6).

The middleware is route-aware (``request.url.path`` is matched against a
template so the cardinality of the ``route`` label stays bounded) and
preserves the existing ``X-Request-ID`` + ``Server-Timing`` semantics of
the official API. When ``prometheus-client`` is not installed the
middleware is a no-op so the API keeps serving traffic.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from triage_ml.dev_api.app import ALLOWED_ERROR_CODES, LANGUAGE_ERROR_CODES
from triage_ml.observability.metrics import (
    PREDICTION_ERRORS_TOTAL,
    PROMETHEUS_AVAILABLE,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
)

if TYPE_CHECKING:
    pass


# Templates keep the ``route`` label cardinality bounded. Routes that take a
# dynamic path parameter (``/reload/{model_version}``) are normalised so the
# metric does not leak the model version into the time series.
_ROUTE_TEMPLATES: tuple[tuple[str, str], ...] = (
    (r"^/predict/?$", "/predict"),
    (r"^/reload/?$", "/reload"),
    (r"^/health/?$", "/health"),
    (r"^/model-info/?$", "/model-info"),
    (r"^/models/?$", "/models"),
    (r"^/metrics/?$", "/metrics"),
    (r"^/$", "/"),
)

# Allow-list of public ``error_code`` values that may be used as a metric
# label. Mirrors the values declared in
# ``triage_ml.dev_api.app.ALLOWED_ERROR_CODES`` (single source of truth) so
# the dashboard never leaks an internal ``detail`` string (privacy
# regression test #1). ``request_failed`` is added as the HTTPException
# fallback.
_METRIC_ERROR_CODES: frozenset[str] = frozenset(
    ALLOWED_ERROR_CODES | LANGUAGE_ERROR_CODES | {"request_failed"}
)


def _normalise_error_code(value: object) -> str | None:
    """Validate ``error_code`` against the public allow-list.

    Returns the normalised code (``strip()`` + ``lower()``) or ``None``
    when the value falls outside the allow-list (intentionally dropped,
    never silently coerced to a fake label).
    """

    if value is None:
        return None
    candidate = str(value).strip().lower()
    if not candidate:
        return None
    if candidate not in _METRIC_ERROR_CODES:
        return None
    return candidate


def _normalise_route(path: str) -> str:
    for pattern, template in _ROUTE_TEMPLATES:
        if re.match(pattern, path):
            return template
    return "/other"


def _model_variant(request: Request) -> str:
    """Return the active variant label recorded alongside metrics.

    The middleware reads ``app.state.model_variant`` set by the lifespan
    handler; the fallback to ``"sklearn"`` covers the early life of the
    app (before lifespan has run) and tests with ``TestClient`` without
    the lifespan context. The normalization is single-source: this is the
    only place the value falls back.
    """

    return getattr(request.app.state, "model_variant", "sklearn")


class PrometheusMiddleware:
    """ASGI middleware that emits Prometheus metrics for every request."""

    def __init__(self, app: Callable[[Request, Callable], Awaitable[Response]]) -> None:
        self._app = app

    async def __call__(
        self,
        scope: dict[str, object],
        receive: Callable[[], Awaitable[dict]],
        send: Callable[[dict], Awaitable[None]],
    ) -> None:
        if not PROMETHEUS_AVAILABLE or scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        start = time.perf_counter()
        status_holder: dict[str, int] = {"status": 500}

        async def _send(message: dict[str, object]) -> None:
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 500))
            await send(message)

        try:
            await self._app(scope, receive, _send)
        finally:
            route = _normalise_route(request.url.path)
            method = request.method.upper()
            variant = _model_variant(request)
            elapsed = time.perf_counter() - start
            REQUESTS_TOTAL.labels(
                route=route,
                method=method,
                status=str(status_holder["status"]),
                model_variant=variant,
            ).inc()
            REQUEST_LATENCY_SECONDS.labels(
                route=route, method=method, model_variant=variant
            ).observe(elapsed)
            error_code_raw = getattr(request.state, "error_code", None)
            error_code = _normalise_error_code(error_code_raw)
            if (
                status_holder["status"] >= 400
                and error_code
                and PREDICTION_ERRORS_TOTAL is not None
            ):
                # ``_normalise_error_code`` returns ``None`` when the value
                # is outside the public allow-list; in that case we drop
                # the increment silently rather than pollute the
                # dashboard with arbitrary strings (hardening #19).
                PREDICTION_ERRORS_TOTAL.labels(
                    route=route, error_code=error_code, model_variant=variant
                ).inc()

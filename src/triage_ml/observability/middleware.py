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


def _normalise_route(path: str) -> str:
    for pattern, template in _ROUTE_TEMPLATES:
        if re.match(pattern, path):
            return template
    return "/other"


def _model_variant(request: Request) -> str:
    """Return the active variant label recorded alongside metrics.

    The middleware lives before ``create_app`` mounts the lifespan handler,
    so the value falls back to ``"sklearn"`` when no variant has been
    resolved yet. ``app.state.model_variant`` is set during startup so the
    real value lands on the first business request.
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
            error_code = getattr(request.state, "error_code", None)
            if (
                status_holder["status"] >= 400
                and error_code
                and PREDICTION_ERRORS_TOTAL is not None
            ):
                PREDICTION_ERRORS_TOTAL.labels(
                    route=route, error_code=str(error_code), model_variant=variant
                ).inc()

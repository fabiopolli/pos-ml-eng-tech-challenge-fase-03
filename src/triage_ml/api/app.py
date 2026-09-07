"""Production FastAPI serving the triage model with RBAC and Observability."""

import os
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from triage_ml.api.auth import RequirePredictRole, RequireRole
from triage_ml.api.logging_config import setup_logging
from triage_ml.api.ratelimit import create_limiters
from triage_ml.api.schemas import (
    ErrorOut,
    HealthOut,
    ModelInfoOut,
    ModelsListOut,
    PredictIn,
    PredictOut,
    ReloadIn,
    ReloadOut,
)
from triage_ml.api.settings import Settings, get_settings
from triage_ml.dev_api.app import (
    ALLOWED_ERROR_CODES,
    ModelHolder,
    _default_model_path,
    _list_model_versions,
)
from triage_ml.dev_api.config import get_api_config
from triage_ml.dev_api.language import UnsupportedLanguageError, detect_language

logger = structlog.get_logger("triage_ml.api")


def _resolve_error_code(detail: object) -> str:
    """Map ``HTTPException.detail`` to a controlled allow-list.

    Mirrors ``triage_ml.dev_api.app`` so prod and dev return the same
    error_code vocabulary. Unknown or non-string details collapse to
    ``"request_failed"`` so internal exception messages never leak into
    the response body.
    """

    if isinstance(detail, str) and detail in ALLOWED_ERROR_CODES:
        return detail
    return "request_failed"


def _request_id_for(request: Request) -> str | None:
    """Return the per-request correlation id or ``None`` if the middleware
    did not run (defensive: the middleware is always mounted, but a stray
    unhandled exception before it should not surface ``"unknown"`` to the
    client)."""

    return getattr(request.state, "request_id", None)


def create_app(*, holder: ModelHolder | None = None, settings: Settings | None = None) -> FastAPI:
    """Build an isolated production API instance.

    ``settings`` is injectable so tests do not depend on process environment
    variables or share rate-limit state with the module-level ASGI app.
    """

    settings = settings or get_settings()
    setup_logging(settings.log_level)

    if holder is None:
        configured_path = os.environ.get("MODEL_PATH") or _default_model_path()
        holder = ModelHolder(configured_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        holder.load()
        yield

    app = FastAPI(title="Triage ML - Prod API", lifespan=lifespan)
    ip_limiter, api_key_limiter = create_limiters()
    app.state.limiter = ip_limiter
    app.state.api_key_limiter = api_key_limiter

    @app.middleware("http")
    async def trace_and_timing_middleware(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start_time = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id

        # Server-Timing: always emit the total, then enrich with per-stage
        # measurements when the handler populated ``request.state``. This
        # keeps observability uniform across endpoints that do not need
        # language detection (e.g. /health, /models, /reload).
        detect_ms = getattr(request.state, "detect_latency_ms", None)
        predict_ms = getattr(request.state, "predict_latency_ms", None)
        timing_parts = [f"total;dur={latency_ms:.3f}"]
        if detect_ms is not None:
            timing_parts.append(f"detect;dur={detect_ms:.3f}")
        if predict_ms is not None:
            timing_parts.append(f"predict;dur={predict_ms:.3f}")
        response.headers["Server-Timing"] = ", ".join(timing_parts)

        # Logging sanitizado (nunca exibe payloads clínicos)
        logger.info("request_finished", status_code=response.status_code, latency_ms=latency_ms)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Log the failing fields server-side for observability without ever
        # echoing them to the client (avoids leaking the input shape).
        req_id = _request_id_for(request)
        structlog.contextvars.bind_contextvars(request_id=req_id)
        logger.info(
            "validation_failed",
            error_count=len(exc.errors()),
            error_types=[err.get("type") for err in exc.errors()],
        )
        return JSONResponse(
            status_code=422,
            content=ErrorOut(
                request_id=req_id,
                error_code="validation_failed",
                message="Request body is invalid.",
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        req_id = _request_id_for(request)
        error_code = _resolve_error_code(exc.detail)
        if isinstance(exc.detail, str) and exc.detail != error_code:
            # Detail was a string but not in the allow-list; record it for
            # diagnostics without echoing it to the client.
            logger.warning(
                "error_code_filtered",
                requested_code=exc.detail,
                resolved_code=error_code,
                status_code=exc.status_code,
            )

        message = "Request could not be processed."
        if error_code == "clinician_review_required":
            message = "Patient roles cannot access raw clinical predictions directly."
        elif error_code == "unauthorized":
            message = "Missing or invalid API Key."
        elif error_code == "forbidden":
            message = "API key is not authorized for this operation."

        # Propaga dados de política de idioma sem vazar o texto
        det_lang = getattr(request.state, "detected_language", None)
        det_score = getattr(request.state, "detected_language_score", None)

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorOut(
                request_id=req_id,
                error_code=error_code,
                message=message,
                detected_language=det_lang,
                detected_language_score=det_score,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_handler(request: Request, exc: Exception):
        req_id = _request_id_for(request)
        # ``logger.exception`` rides on ``format_exc_info`` from
        # ``logging_config.setup_logging`` to surface the traceback.
        logger.exception("internal_error", error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=ErrorOut(
                request_id=req_id, error_code="internal_error", message="Internal Server Error."
            ).model_dump(),
        )

    # =========================================================================
    # Observabilidade (Abertos sem Auth, conforme PLAN-api-prod.md)
    # =========================================================================

    @app.get("/health", response_model=HealthOut)
    def health(request: Request):
        _, _, _, model_version = holder.snapshot()
        return HealthOut(
            status="ok" if holder.loaded else "degraded",
            model_version=model_version,
            model_loaded=holder.loaded,
        )

    @app.get("/model-info", response_model=ModelInfoOut)
    def model_info(request: Request, role: str = Depends(RequireRole(["service", "doctor"]))):
        # ``/model-info`` exposes the full training manifest (label names,
        # hyperparameters, dependency versions, git metadata). Restrict it
        # to service and doctor roles so unauthenticated probes cannot
        # fingerprint the stack or the class taxonomy.
        pipeline, metadata, _, _ = holder.snapshot()
        if not pipeline:
            raise HTTPException(status_code=503, detail="model_not_ready")
        return ModelInfoOut(**metadata)

    @app.get("/models", response_model=ModelsListOut)
    def list_models(
        request: Request, role: str = Depends(RequireRole(["service", "doctor"]))
    ):
        # Same reasoning as /model-info: the registry fingerprint is useful
        # reconnaissance data and must stay behind RBAC.
        _, _, _, model_version = holder.snapshot()
        return ModelsListOut(
            versions=_list_model_versions(holder.registry_root), current=model_version
        )

    # =========================================================================
    # Operações Controladas (Requerem RBAC via API Key)
    # =========================================================================

    @app.post("/reload", response_model=ReloadOut)
    @ip_limiter.limit(settings.ratelimit_default)
    @api_key_limiter.limit(settings.ratelimit_default)
    def reload_model(
        request: Request,
        payload: ReloadIn,
        role: str = Depends(RequireRole(["service"])),
    ):
        try:
            version = holder.reload_to(payload.model_version)
            logger.info("model_reloaded", model_version=version, role=role)
            return ReloadOut(model_version=version, model_loaded=True)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="model_not_found") from exc
        except Exception as exc:
            logger.error(
                "reload_failed",
                model_version=payload.model_version,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail="model_incompatible") from exc

    @app.post("/predict", response_model=PredictOut)
    @ip_limiter.limit(settings.ratelimit_predict)
    @api_key_limiter.limit(settings.ratelimit_predict)
    def predict(
        request: Request,
        role: str = Depends(RequirePredictRole()),
        payload: PredictIn = ...,
    ):
        # Security Gate: RBAC is centralised in ``RequirePredictRole``.
        # ``doctor`` is the only role allowed to invoke /predict; patient
        # is rejected with ``clinician_review_required`` (different from
        # the generic ``forbidden`` so dashboards can render a meaningful
        # explanation), and service is rejected with ``forbidden``.
        _ = role  # already enforced by RequirePredictRole above

        req_id = _request_id_for(request)
        pipeline, metadata, label_names, model_version = holder.snapshot()
        if not pipeline:
            raise HTTPException(status_code=503, detail="model_not_ready")

        api_config = get_api_config()

        # Etapa de verificação de idioma
        detect_start = time.perf_counter()
        try:
            detect_language(
                payload.text,
                min_chars=api_config.min_text_chars_for_language_check,
                min_score=api_config.min_language_score,
                supported=api_config.supported_languages,
            )
        except UnsupportedLanguageError as exc:
            request.state.detect_latency_ms = (time.perf_counter() - detect_start) * 1000.0
            request.state.detected_language = exc.code
            request.state.detected_language_score = exc.score
            raise HTTPException(status_code=422, detail=exc.reason) from exc

        request.state.detect_latency_ms = (time.perf_counter() - detect_start) * 1000.0

        # Etapa de predição — ``predict_latency_ms`` é medido a partir daqui
        # para que o campo ``latency_ms`` no body cubra apenas a inferência
        # do pipeline (e não a detecção de idioma, que já tem seu próprio
        # slot em ``Server-Timing``).
        predict_start = time.perf_counter()
        try:
            label = int(pipeline.predict([payload.text])[0])
            score = None
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba([payload.text])[0]
                index = list(pipeline.classes_).index(label)
                score = float(proba[index])
        except Exception as exc:
            request.state.predict_latency_ms = (time.perf_counter() - predict_start) * 1000.0
            logger.error(
                "prediction_failed",
                error_type=type(exc).__name__,
                model_version=model_version,
            )
            raise HTTPException(status_code=500, detail="prediction_failed") from exc

        request.state.predict_latency_ms = (time.perf_counter() - predict_start) * 1000.0

        latency = (time.perf_counter() - predict_start) * 1000
        return PredictOut(
            label=label,
            label_name=label_names.get(label, str(label)),
            score=score,
            model_version=model_version or "unknown",
            latency_ms=round(latency, 3),
            request_id=req_id or "unknown",
        )

    return app


# Exporta a instância ASGI para o uvicorn (Ex: uvicorn triage_ml.api.app:app)
app = create_app()

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

from triage_ml.api.auth import RequireRole, get_current_role
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
from triage_ml.dev_api.app import ModelHolder, _default_model_path, _list_model_versions
from triage_ml.dev_api.config import get_api_config
from triage_ml.dev_api.language import UnsupportedLanguageError, detect_language

logger = structlog.get_logger("triage_ml.api")


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

        # Headers Server-Timing para observabilidade
        detect_ms = getattr(request.state, "detect_latency_ms", None)
        predict_ms = getattr(request.state, "predict_latency_ms", None)
        timing_parts = []
        if detect_ms is not None:
            timing_parts.append(f"detect;dur={detect_ms:.3f}")
        if predict_ms is not None:
            timing_parts.append(f"predict;dur={predict_ms:.3f}")
        if timing_parts:
            response.headers["Server-Timing"] = ", ".join(timing_parts)

        # Logging sanitizado (nunca exibe payloads clínicos)
        logger.info("request_finished", status_code=response.status_code, latency_ms=latency_ms)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=422,
            content=ErrorOut(
                request_id=req_id, error_code="validation_failed", message="Invalid payload."
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        req_id = getattr(request.state, "request_id", "unknown")
        error_code = exc.detail if isinstance(exc.detail, str) else "request_failed"
        message = "Request could not be processed."

        if error_code == "clinician_review_required":
            message = "Patient roles cannot access raw clinical predictions directly."
        elif error_code == "unauthorized":
            message = "Missing or invalid API Key."

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
        req_id = getattr(request.state, "request_id", "unknown")
        # Sanitização de log: captura apenas o tipo da exceção, evita expor o texto
        logger.error("internal_error", error_type=type(exc).__name__)
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
    def model_info(request: Request):
        pipeline, metadata, _, _ = holder.snapshot()
        if not pipeline:
            raise HTTPException(status_code=503, detail="model_not_ready")
        return ModelInfoOut(**metadata)

    @app.get("/models", response_model=ModelsListOut)
    def list_models(request: Request):
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
        request: Request, payload: ReloadIn, role: str = Depends(RequireRole(["service"]))
    ):
        try:
            version = holder.reload_to(payload.model_version)
            return ReloadOut(model_version=version, model_loaded=True)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="model_not_found") from exc
        except Exception as exc:
            logger.error("reload_failed", error_type=type(exc).__name__)
            raise HTTPException(status_code=500, detail="model_incompatible") from exc

    @app.post("/predict", response_model=PredictOut)
    @ip_limiter.limit(settings.ratelimit_predict)
    @api_key_limiter.limit(settings.ratelimit_predict)
    def predict(request: Request, payload: PredictIn, role: str = Depends(get_current_role)):
        # Security Gate: Strict RBAC
        if role == "patient":
            raise HTTPException(status_code=403, detail="clinician_review_required")
        if role != "doctor":
            raise HTTPException(status_code=403, detail="forbidden")

        start = time.perf_counter()
        req_id = request.state.request_id

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

        # Etapa de predição
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
            logger.error("prediction_failed", error_type=type(exc).__name__)
            raise HTTPException(status_code=500, detail="prediction_failed") from exc

        request.state.predict_latency_ms = (time.perf_counter() - predict_start) * 1000.0

        latency = (time.perf_counter() - start) * 1000
        return PredictOut(
            label=label,
            label_name=label_names.get(label, str(label)),
            score=score,
            model_version=model_version or "unknown",
            latency_ms=latency,
            request_id=req_id,
        )

    return app


# Exporta a instância ASGI para o uvicorn (Ex: uvicorn triage_ml.api.app:app)
app = create_app()

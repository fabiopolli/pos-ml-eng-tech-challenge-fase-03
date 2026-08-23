"""Local smoke API for a validated baseline model artifact."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from triage_ml.api.schemas import ErrorOut, HealthOut, PredictIn, PredictOut
from triage_ml.models.artifact import load_artifact

logger = logging.getLogger("triage_ml.api")
REPO_ROOT = Path(__file__).resolve().parents[3]
VERSION_DIR_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{12}$")


def _default_model_path() -> Path:
    models_dir = REPO_ROOT / "models"
    versioned = (
        sorted(
            (
                path / "model.joblib"
                for path in models_dir.iterdir()
                if path.is_dir() and VERSION_DIR_PATTERN.fullmatch(path.name)
            ),
            reverse=True,
        )
        if models_dir.exists()
        else []
    )
    if versioned:
        return versioned[0]
    return models_dir / "v1" / "model.joblib"


class ModelHolder:
    """Load and publish a model only after full artifact validation."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.pipeline: Any = None
        self.metadata: dict[str, Any] = {}
        self.label_names: dict[int, str] = {}
        self.model_version: str | None = None

    def load(self) -> None:
        self.pipeline = None
        self.metadata = {}
        self.label_names = {}
        self.model_version = None
        try:
            pipeline, metadata = load_artifact(self.model_path)
        except Exception as exc:
            logger.error("model startup validation failed")
            raise RuntimeError("model artifact is missing or incompatible") from exc
        self.metadata = metadata
        self.label_names = {int(label): name for label, name in metadata["label_mapping"].items()}
        self.model_version = metadata["model_version"]
        self.pipeline = pipeline

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None


def _request_id() -> str:
    """Generate an internal ID so client-controlled content never reaches logs."""

    return uuid.uuid4().hex[:12]


def create_app(
    *,
    model_path: str | Path | None = None,
    holder: ModelHolder | None = None,
) -> FastAPI:
    """Create the app with an injectable holder for hermetic tests."""

    if holder is None:
        configured_path = model_path or os.environ.get("MODEL_PATH") or _default_model_path()
        holder = ModelHolder(configured_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        holder.load()
        yield

    app = FastAPI(title="triage_ml smoke API", lifespan=lifespan)
    app.state.holder = holder

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = _request_id()
        request.state.started_at = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        latency_ms = getattr(request.state, "predict_latency_ms", None)
        if latency_ms is not None:
            response.headers["Server-Timing"] = f"predict;dur={latency_ms:.3f}"
        return response

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, _: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        body = ErrorOut(
            request_id=request_id,
            error_code="validation_failed",
            message="Request body is invalid.",
        ).model_dump()
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=body)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", None)
        allowed_codes = {"validation_failed", "prediction_failed", "model_not_ready"}
        error_code = exc.detail if exc.detail in allowed_codes else "request_failed"
        body = ErrorOut(
            request_id=request_id,
            error_code=error_code,
            message="Request could not be processed.",
        ).model_dump()
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, _: Exception):
        request_id = getattr(request.state, "request_id", None)
        latency_ms = (time.perf_counter() - request.state.started_at) * 1000.0
        logger.error(
            "request failed (rid=%s route=%s status=500 latency_ms=%.3f)",
            request_id,
            request.url.path,
            latency_ms,
        )
        body = ErrorOut(
            request_id=request_id,
            error_code="internal_error",
            message="Request could not be processed.",
        ).model_dump()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body)

    @app.get("/health", response_model=HealthOut)
    async def health() -> HealthOut:
        return HealthOut(
            status="ok" if holder.loaded else "degraded",
            model_version=holder.model_version,
            model_loaded=holder.loaded,
        )

    @app.post("/predict", response_model=PredictOut)
    async def predict(payload: PredictIn, request: Request) -> PredictOut:
        request_id: str = request.state.request_id
        if not holder.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model_not_ready",
            )

        text = payload.text
        if not text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="validation_failed",
            )
        warnings: list[str] = []

        started = time.perf_counter()
        try:
            label = int(holder.pipeline.predict([text])[0])
            score: float | None = None
            if hasattr(holder.pipeline, "predict_proba"):
                proba = holder.pipeline.predict_proba([text])[0]
                index = list(holder.pipeline.classes_).index(label)
                score = float(proba[index])
        except Exception as exc:
            request.state.predict_latency_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "prediction failed (rid=%s route=/predict status=500 latency_ms=%.3f)",
                request_id,
                request.state.predict_latency_ms,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="prediction_failed",
            ) from exc
        else:
            request.state.predict_latency_ms = (time.perf_counter() - started) * 1000.0

        return PredictOut(
            label=label,
            label_name=holder.label_names[label],
            score=score,
            model_version=holder.model_version or "unknown",
            latency_ms=request.state.predict_latency_ms,
            request_id=request_id,
            warnings=warnings,
        )

    return app


app = create_app()

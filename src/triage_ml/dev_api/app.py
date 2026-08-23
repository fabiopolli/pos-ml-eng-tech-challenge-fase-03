"""Local dev API for a validated baseline model artifact.

The ``dev_api`` package is the developer-facing FastAPI app: it consumes
the real trained artifact from ``models/`` (TF-IDF + LinearSVC) and is
used by Bill (and any contributor) to validate the pipeline,
serialization, language policy and observability primitives locally
before the official API from Romário (Etapa 3 of the checklist) ships.

It is **not** a stub, **not** a fake, and **not** production-ready: it
has no auth, no rate limit, no Prometheus instrumentation and no
Docker packaging. Treat it as a thin, well-tested seam around the
artifact contract.
"""

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

from triage_ml.dev_api.config import get_api_config
from triage_ml.dev_api.language import UnsupportedLanguageError, detect_language
from triage_ml.dev_api.schemas import (
    ErrorOut,
    HealthOut,
    ModelInfoOut,
    ModelsListOut,
    PredictIn,
    PredictOut,
    ReloadIn,
    ReloadOut,
)
from triage_ml.models.artifact import load_artifact

logger = logging.getLogger("triage_ml.dev_api")
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


def _list_model_versions() -> list[str]:
    """Return all immutable artifact versions available under ``models/``.

    The listing is ordered newest-first (matching ``_default_model_path``)
    and filters out the legacy ``v1/`` directory that no longer satisfies
    the validated manifest contract. Useful for the dashboard's
    model-picker and for the ``GET /models`` endpoint.
    """

    models_dir = REPO_ROOT / "models"
    if not models_dir.exists():
        return []
    return sorted(
        (
            path.name
            for path in models_dir.iterdir()
            if path.is_dir() and VERSION_DIR_PATTERN.fullmatch(path.name)
        ),
        reverse=True,
    )


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

    def reload_to(self, version: str) -> None:
        """Atomically swap the holder to a different immutable artifact version.

        The new path is resolved against the ``models/`` directory and
        must obey ``VERSION_DIR_PATTERN``; ``load_artifact`` re-runs the
        full validation (manifest, checksum, classes) before the swap
        actually commits, so the holder never observes a half-loaded
        state. Raises ``FileNotFoundError`` if the version is unknown
        and propagates any ``RuntimeError`` from ``load_artifact`` when
        the artifact is incompatible.
        """

        if not VERSION_DIR_PATTERN.fullmatch(version):
            raise FileNotFoundError(f"unknown model version: {version!r}")
        new_path = REPO_ROOT / "models" / version / "model.joblib"
        if not new_path.is_file():
            raise FileNotFoundError(f"unknown model version: {version!r}")
        # ``load_artifact`` is responsible for the validation; on success
        # it returns the pipeline + metadata we need to publish.
        pipeline, metadata = load_artifact(new_path)
        self.pipeline = pipeline
        self.metadata = metadata
        self.label_names = {int(label): name for label, name in metadata["label_mapping"].items()}
        self.model_version = metadata["model_version"]
        self.model_path = new_path

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

    app = FastAPI(title="triage_ml dev API", lifespan=lifespan)
    app.state.holder = holder

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = _request_id()
        request.state.started_at = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        detect_ms = getattr(request.state, "detect_latency_ms", None)
        predict_ms = getattr(request.state, "predict_latency_ms", None)
        timing_parts: list[str] = []
        if detect_ms is not None:
            timing_parts.append(f"detect;dur={detect_ms:.3f}")
        if predict_ms is not None:
            timing_parts.append(f"predict;dur={predict_ms:.3f}")
        if timing_parts:
            response.headers["Server-Timing"] = ", ".join(timing_parts)
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
        language_codes = {
            "unsupported_language",
            "text_too_short_for_language_check",
            "indeterminate_language",
        }
        allowed_codes = {
            "validation_failed",
            "prediction_failed",
            "model_not_ready",
            "model_not_found",
            "model_incompatible",
        } | language_codes
        error_code = exc.detail if exc.detail in allowed_codes else "request_failed"
        message = "Request could not be processed."
        if error_code in language_codes:
            message = "Only English texts are supported."
        elif error_code == "model_not_found":
            message = "Requested model version was not found under models/."
        detected_language: str | None = None
        detected_language_score: float | None = None
        # Attach the detector's verdict when the route populated it.
        last_check = getattr(request.state, "last_language_check", None)
        if last_check is not None:
            detected_language = last_check.code
            detected_language_score = last_check.score
        body = ErrorOut(
            request_id=request_id,
            error_code=error_code,
            message=message,
            detected_language=detected_language,
            detected_language_score=detected_language_score,
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

    @app.get("/models", response_model=ModelsListOut)
    async def list_models() -> ModelsListOut:
        """List every immutable artifact version available under ``models/``.

        Pure read-only endpoint — does not touch the loaded pipeline.
        Returns the version list newest-first and echoes the currently
        loaded version so the dashboard can preselect it in a picker.
        """

        return ModelsListOut(versions=_list_model_versions(), current=holder.model_version)

    @app.get("/model-info", response_model=ModelInfoOut)
    async def model_info() -> ModelInfoOut:
        """Expose the validated artifact manifest.

        The endpoint serves the same ``metadata.json`` content the API
        loaded at startup. Callers can inspect metrics, training split
        sizes, classifier selection details and dependency versions
        without having to read the filesystem directly. Returns
        ``503 model_not_ready`` when the model is not loaded.
        """

        if not holder.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model_not_ready",
            )
        return ModelInfoOut(**holder.metadata)

    @app.post("/reload", response_model=ReloadOut)
    async def reload_model(payload: ReloadIn) -> ReloadOut:
        """Swap the holder to a different validated artifact version.

        Body: ``{"model_version": "YYYYMMDDTHHMMSSZ-<12hex>"}``. The new
        artifact must already exist under ``models/<version>/``; the
        reload re-runs ``load_artifact`` so a failed validation aborts
        before the holder is mutated.

        Errors:

        * ``404 model_not_found`` — version is unknown or malformed.
        * ``500 model_incompatible`` — manifest, checksum or class
          mismatch on the requested version; the previous model stays
          loaded (the swap is rejected, not partially applied).
        """

        version = payload.model_version
        try:
            holder.reload_to(version)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="model_not_found",
            ) from exc
        except (
            RuntimeError,
            Exception,
        ) as exc:  # ``load_artifact`` raises multiple exception types
            logger.error("model reload to %s failed: %s", version, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="model_incompatible",
            ) from exc
        logger.info("model reload to %s succeeded", holder.model_version)
        return ReloadOut(model_version=holder.model_version or "", model_loaded=holder.loaded)

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

        api_config = get_api_config()
        detect_started = time.perf_counter()
        try:
            detect_language(
                text,
                min_chars=api_config.min_text_chars_for_language_check,
                min_score=api_config.min_language_score,
                supported=api_config.supported_languages,
            )
        except UnsupportedLanguageError as exc:
            detect_latency_ms = (time.perf_counter() - detect_started) * 1000.0
            request.state.detect_latency_ms = detect_latency_ms
            request.state.last_language_check = exc
            logger.info(
                "language check rejected (rid=%s reason=%s code=%s score=%s detect_ms=%.3f)",
                request_id,
                exc.reason,
                exc.code,
                exc.score,
                detect_latency_ms,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=exc.reason,
            ) from exc
        else:
            request.state.detect_latency_ms = (time.perf_counter() - detect_started) * 1000.0
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

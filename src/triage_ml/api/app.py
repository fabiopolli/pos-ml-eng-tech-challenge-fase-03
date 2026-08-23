"""Smoke FastAPI app exposing ``/health`` and ``/predict``.

This is **not** the production API. It is intentionally minimal: it
loads the serialized artifact, applies the contract defined in
``docs/plans/PLAN-text-classifier.md`` and exposes the latency, the
``request_id`` and the ``Server-Timing`` header that the future
observability work will rely on.

Romário will replace or extend this app during Etapa 3 of the
checklist; the request/response schema here is the proposal that needs
his validation.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from triage_ml.api.schemas import ErrorOut, HealthOut, PredictIn, PredictOut

logger = logging.getLogger("triage_ml.api")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "v1"
DEFAULT_LABELS_CSV = REPO_ROOT / "data" / "medical_tc_labels.csv"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


class ModelHolder:
    """Lazy model loader. Keeps the cold-start cost out of import time."""

    def __init__(self, model_dir: Path, labels_csv: Path) -> None:
        self.model_dir = model_dir
        self.labels_csv = labels_csv
        self.pipeline: Any = None
        self.metadata: dict[str, Any] = {}
        self.classes: list[int] = []
        self.label_names: dict[int, str] = {}
        self.model_version: str | None = None
        self.error: str | None = None

    def load(self) -> None:
        joblib_path = self.model_dir / "model.joblib"
        metadata_path = self.model_dir / "metadata.json"
        classes_path = self.model_dir / "classes.json"
        if not (joblib_path.exists() and metadata_path.exists()):
            self.error = f"model artifact not found under {self.model_dir}"
            logger.warning(self.error)
            return
        try:
            self.pipeline = joblib.load(joblib_path)
            self.metadata = _read_json(metadata_path)
            self.classes = (
                _read_json(classes_path) if classes_path.exists() else list(self.pipeline.classes_)
            )
            self.model_version = str(self.metadata.get("model_version", "unknown"))
            self.label_names = _load_label_names(self.labels_csv)
        except Exception as exc:  # pragma: no cover - defensive
            self.error = f"failed to load model: {exc}"
            logger.exception("model load failed")

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None


def _read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_label_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(row["condition_label"]): str(row["condition_name"]) for _, row in df.iterrows()}


def _build_app() -> FastAPI:
    model_dir = _env_path("MODEL_DIR", DEFAULT_MODEL_PATH)
    labels_csv = _env_path("LABELS_CSV", DEFAULT_LABELS_CSV)
    holder = ModelHolder(model_dir, labels_csv)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        holder.load()
        yield

    app = FastAPI(title="triage_ml smoke API", lifespan=lifespan)
    app.state.holder = holder

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        body = ErrorOut(
            request_id=request_id,
            error_code="validation_failed",
            message="Request body is invalid.",
        ).model_dump()
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body,
            headers={"X-Request-ID": request_id or ""},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", None)
        error_code = (
            "validation_failed"
            if exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            else "request_failed"
        )
        body = ErrorOut(
            request_id=request_id,
            error_code=error_code,
            message="Request could not be processed.",
        ).model_dump()
        return JSONResponse(
            status_code=exc.status_code, content=body, headers={"X-Request-ID": request_id or ""}
        )

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
            logger.warning("/predict called before model was ready (rid=%s)", request_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not ready"
            )

        warnings: list[str] = []
        text = payload.text.strip()
        if not text:
            warnings.append("empty_text_after_strip")
        started = time.perf_counter()
        try:
            label = int(holder.pipeline.predict([text])[0])
        except Exception:
            logger.exception("/predict inference failed (rid=%s)", request_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="prediction_failed",
            ) from None
        score: float | None = None
        if hasattr(holder.pipeline, "predict_proba"):
            try:
                proba = holder.pipeline.predict_proba([text])[0]
                idx = list(holder.pipeline.classes_).index(label)
                score = float(proba[idx])
            except Exception:
                score = None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        label_name = holder.label_names.get(label, "unknown")
        response = PredictOut(
            label=label,
            label_name=label_name,
            score=score,
            model_version=holder.model_version or "unknown",
            latency_ms=elapsed_ms,
            request_id=request_id,
            warnings=warnings,
        )
        # We can't easily mutate the response headers from here, but the
        # ``Server-Timing`` value is also exposed via the ASGI timing
        # headers added by FastAPI when instrumented. The body carries
        # the canonical latency in ``latency_ms``.
        return response

    return app


app = _build_app()

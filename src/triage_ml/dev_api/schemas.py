"""Pydantic schemas for the dev API.

These schemas are the initial proposal documented in
``docs/plans/PLAN-text-classifier.md``. They are **subject to validation
by Romário** before being promoted as the contract of the production
API.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

# Field bound rejects oversized text after JSON parsing while still fitting
# the longest abstract in the corpus. Deployment must limit HTTP body bytes.
# (median ~1.2k characters, max ~4k).


class PredictIn(BaseModel):
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=20000),
    ] = Field(
        description="Free-text medical abstract to classify.",
    )


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None = None
    model_loaded: bool


class ModelInfoOut(BaseModel):
    """Snapshot of the validated artifact manifest.

    Returned by ``GET /model-info`` so that callers (dashboard, smoke
    script, monitoring) can inspect what the API is currently serving
    without having to read the filesystem directly. The shape mirrors
    the canonical ``metadata.json`` schema, restricted to the keys the
    manifest contract requires (see
    ``src/triage_ml/models/artifact.py::REQUIRED_METADATA_KEYS``).

    The nested dicts (``metrics``, ``preprocessing``, ``selection``)
    are typed permissively because their inner shape is already
    enforced by ``validate_metadata`` when the artifact was loaded.
    """

    model_version: str
    model_name: str
    task_type: str
    language: str
    classes: list[int]
    label_mapping: dict[str, str]
    random_state: int
    n_train: int
    n_test: int
    metrics: dict[str, Any]
    preprocessing: dict[str, Any]
    selection: dict[str, Any]
    dependency_versions: dict[str, str]
    git_commit: str
    git_dirty: bool
    created_at: str


class PredictOut(BaseModel):
    label: int
    label_name: str
    score: float | None = None
    model_version: str
    latency_ms: float
    request_id: str
    warnings: list[str] = Field(default_factory=list)


class ErrorOut(BaseModel):
    request_id: str | None = None
    error_code: str
    message: str
    detected_language: str | None = None
    detected_language_score: float | None = None


class ReloadIn(BaseModel):
    """Body for ``POST /reload``.

    ``model_version`` must match the immutable schema
    ``YYYYMMDDTHHMMSSZ-<12 hex>`` and correspond to a directory under
    ``models/`` containing a validated ``model.joblib`` + ``metadata.json``.
    """

    model_version: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
    ]


class ReloadOut(BaseModel):
    """Response for ``POST /reload``.

    Mirrors ``HealthOut`` so the dashboard can refresh both blocks from
    the same payload. ``model_version`` echoes the validated version
    (may differ from the requested one when the artifact directory name
    does not match ``metadata.model_version``).
    """

    model_version: str
    model_loaded: bool


class ModelsListOut(BaseModel):
    """Response for ``GET /models``.

    Lists every immutable artifact version discovered under ``models/``,
    newest first. Excludes the legacy ``v1/`` directory.
    """

    versions: list[str]
    current: str | None = None

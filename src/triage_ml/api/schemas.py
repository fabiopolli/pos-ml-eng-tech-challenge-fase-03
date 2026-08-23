"""Pydantic schemas for the smoke API.

These schemas are the initial proposal documented in
``docs/plans/PLAN-text-classifier.md``. They are **subject to validation
by Romário** before being promoted as the contract of the production
API.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# Hard upper bound keeps the request body small and prevents accidental
# abuse while still fitting the longest abstract in the corpus
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

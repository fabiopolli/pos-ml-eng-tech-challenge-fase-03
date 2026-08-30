"""Pydantic schemas for the production API.

This is the canonical source of truth for the HTTP contract.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class PredictIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=20000),
    ] = Field(description="Free-text medical abstract to classify.")


class MetricsPerClassOut(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class MetricsOut(BaseModel):
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, MetricsPerClassOut] = Field(default_factory=dict)


class SelectionCandidateOut(BaseModel):
    fold_macro_f1: list[float]
    mean_macro_f1: float
    std_macro_f1: float


class SelectionOut(BaseModel):
    metric: str
    folds: int
    candidates: dict[str, SelectionCandidateOut]
    best_classifier: str
    selected_classifier: str
    selection_policy: str
    test_set_used_for_selection: bool


class PreprocessingOut(BaseModel):
    vectorizer: str
    tfidf: dict[str, Any]
    classifier: str
    classifier_params: dict[str, Any]


class ModelInfoOut(BaseModel):
    model_version: str
    model_name: str
    task_type: str
    language: str
    classes: list[int]
    label_mapping: dict[str, str]
    random_state: int
    n_train: int
    n_test: int
    metrics: MetricsOut
    preprocessing: PreprocessingOut
    selection: SelectionOut
    dependency_versions: dict[str, str]
    git_commit: str
    git_dirty: bool
    created_at: str


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


class ReloadIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)
    ]


class ReloadOut(BaseModel):
    model_version: str
    model_loaded: bool


class ModelsListOut(BaseModel):
    versions: list[str]
    current: str | None = None

"""Optimization helpers for the baseline text classifier (Fase 2, Etapa 5).

This package contains the building blocks used by the Airflow orchestration
(``triage_ml_retraining_optimization``) and the API official to materialize and
benchmark ONNX variants of the sklearn baseline. ``optimization`` and
``observability`` are independent optional groups; the production API image
does not need the optimization group unless ``TRIAGE_ML_MODEL_VARIANT=onnx``
is requested at runtime.
"""

from triage_ml.optimization.dataloader import DatasetSlice, iter_dataset_slices
from triage_ml.optimization.optimize import OptimizationFingerprint, export_onnx
from triage_ml.optimization.registry import (
    AVAILABLE_VARIANTS,
    resolve_variant_loader,
    validate_variant_metadata,
)

__all__ = [
    "AVAILABLE_VARIANTS",
    "DatasetSlice",
    "OptimizationFingerprint",
    "export_onnx",
    "iter_dataset_slices",
    "resolve_variant_loader",
    "validate_variant_metadata",
]

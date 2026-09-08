"""Variant registry for the optimization pipeline (Fase 2, Etapa 5).

Maps ``MODEL_VARIANT`` env values to the loader that materializes a runtime
"predictor" object from a stored artifact directory. A "predictor" is any
object exposing ``predict``, optional ``predict_proba`` and ``classes_``.

The registry keeps the API official decoupled from the optimization extras:
``resolve_variant_loader("sklearn")`` works without ``onnxruntime`` installed,
and the ONNX loader is only invoked when the operator sets
``TRIAGE_ML_MODEL_VARIANT=onnx`` and the metadata advertises an ONNX variant.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from triage_ml.optimization.onnx_adapter import OnnxModelAdapter

VariantName = Literal["sklearn", "onnx"]
AVAILABLE_VARIANTS: tuple[VariantName, ...] = ("sklearn", "onnx")


def resolve_variant_loader(variant: str) -> Callable[[Path], Any]:
    """Return a function that loads a predictor from ``models/<version>/``.

    The returned loader expects a path to ``model.joblib`` (sklearn) or
    ``model.onnx`` (onnx). Loading is lazy; the registry does not import
    ONNX at registration time.
    """

    if variant == "sklearn":
        return _load_sklearn
    if variant == "onnx":
        return _load_onnx
    raise ValueError(
        f"unknown TRIAGE_ML_MODEL_VARIANT {variant!r}; valid: {list(AVAILABLE_VARIANTS)}"
    )


def _load_sklearn(joblib_path: Path) -> Any:
    import joblib

    return joblib.load(joblib_path)


def _load_onnx(onnx_path: Path) -> OnnxModelAdapter:
    metadata_path = onnx_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = tuple(metadata["classes"])
    preprocessing_classifier = str(metadata.get("preprocessing", {}).get("classifier", "unknown"))
    classifier_kind: Literal["logreg", "linear_svc", "unknown"]
    classifier_kind = (
        preprocessing_classifier
        if preprocessing_classifier in {"logreg", "linear_svc"}
        else "unknown"
    )
    return OnnxModelAdapter(
        onnx_path=onnx_path,
        classes=classes,
        classifier_kind=classifier_kind,
    )


def validate_variant_metadata(metadata: dict[str, Any], *, variant: str) -> None:
    """Raise ``ValueError`` when ``metadata`` does not advertise ``variant``."""

    available = metadata.get("available_variants")
    if not isinstance(available, list) or variant not in available:
        raise ValueError(
            f"metadata does not advertise variant {variant!r}; available: {available!r}"
        )


def read_runtime_variant(default: str = "sklearn") -> VariantName:
    """Resolve the runtime variant from ``TRIAGE_ML_MODEL_VARIANT``."""

    candidate = os.environ.get("TRIAGE_ML_MODEL_VARIANT", default).lower()
    if candidate not in AVAILABLE_VARIANTS:
        raise ValueError(
            f"TRIAGE_ML_MODEL_VARIANT must be one of {AVAILABLE_VARIANTS}; got {candidate!r}"
        )
    return candidate  # type: ignore[return-value]


def resolve_artifact_for_variant(
    model_version_dir: Path,
    variant: str,
    *,
    onnx_filename: str = "model.onnx",
    joblib_filename: str = "model.joblib",
) -> Path:
    """Return the artifact file used for ``variant`` inside ``model_version_dir``."""

    if variant == "sklearn":
        return model_version_dir / joblib_filename
    if variant == "onnx":
        return model_version_dir / onnx_filename
    raise ValueError(f"unknown variant {variant!r}")

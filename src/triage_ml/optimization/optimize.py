"""ONNX export for the baseline TF-IDF + linear classifier pipeline.

The conversion goes through ``skl2onnx.convert_sklearn`` (opset 17) and
preserves the TF-IDF + classifier chain as a single ONNX graph. The
companion adapter (``onnx_adapter``) implements the same ``predict`` /
``predict_proba`` contract as the sklearn pipeline so that the API official
can switch variants without changing the request/response shape.

Dependencies live in the optional ``[optimization]`` group (``pyproject.toml``);
``import skl2onnx`` is deferred to module-level imports guarded by a ``try`` so
that production environments without the optimization group still load the
rest of the package.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised in environments with [optimization]
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by tests in optimization env
    ONNX_AVAILABLE = False


DEFAULT_OPSET = 17


@dataclass(frozen=True)
class OptimizationFingerprint:
    """Stable identifier of an ONNX export configuration.

    The fingerprint ties ``metadata.optimization_fingerprint`` to the exact
    opset/quantization flags used at export time, so the Airflow
    ``find_reusable_artifact`` can reuse a previously materialized variant
    without re-running ``convert_sklearn``.
    """

    classifier: str
    opset: int
    quantized: bool

    def to_dict(self) -> dict[str, Any]:
        return {"classifier": self.classifier, "opset": self.opset, "quantized": self.quantized}


def fingerprint_dict(pipeline: Any, *, opset: int, quantized: bool) -> OptimizationFingerprint:
    """Build the canonical fingerprint for a pipeline + export settings."""

    classifier_name = pipeline.named_steps.get("clf", None)
    classifier_kind = type(classifier_name).__name__ if classifier_name is not None else "unknown"
    return OptimizationFingerprint(
        classifier=classifier_kind, opset=opset, quantized=bool(quantized)
    )


def fingerprint_hash(fingerprint: OptimizationFingerprint) -> str:
    """Stable, short hash of the export fingerprint (used in manifest)."""

    payload = json.dumps(fingerprint.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _initial_type() -> Any:
    """The ONNX "initial type" is one ``string`` column per input row."""

    if not ONNX_AVAILABLE:  # pragma: no cover - import path only
        raise RuntimeError("onnx / skl2onnx are not installed; install the [optimization] group")
    return [("input", StringTensorType([None, 1]))]


def export_onnx(
    pipeline: Any,
    out_path: str | Path,
    *,
    opset: int = DEFAULT_OPSET,
    quantized: bool = False,
    target_opset: int | None = None,
) -> tuple[Path, OptimizationFingerprint]:
    """Serialize ``pipeline`` to an ONNX file with a fingerprint.

    Parameters
    ----------
    pipeline
        A fitted sklearn ``Pipeline`` (``tfidf`` -> ``clf``). The classifier step
        must expose ``classes_`` after fitting.
    out_path
        Destination for ``model.onnx`` (parent directories are created).
    opset
        ONNX opset version. Defaults to ``17`` (supported by ``skl2onnx >= 1.16``
        and ``onnxruntime >= 1.17``).
    quantized
        Reserved for future use; ``skl2onnx`` does not yet ship dynamic-quantization
        primitives for sklearn pipelines and the plan calls for promoting
        quantization only if the unquantized variant fails the latency gate.
    target_opset
        Optional override for the conversion target opset; defaults to
        ``opset`` when ``None``.
    """

    if not ONNX_AVAILABLE:
        raise RuntimeError(
            "triage_ml.optimization: install the [optimization] extra "
            "(pip install -e .[optimization]) before exporting ONNX"
        )
    if not hasattr(pipeline, "named_steps") or set(pipeline.named_steps) != {"tfidf", "clf"}:
        raise ValueError("pipeline must be a fitted sklearn Pipeline('tfidf' -> 'clf')")
    target_opset = target_opset or opset
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    options: dict[str, Any] = {}
    if quantized:
        # Future-proofing: when skl2onnx ships a quantization helper we will
        # forward the relevant options here. For now we warn rather than
        # silently applying an unrelated knob.
        warnings.warn(
            "export_onnx(quantized=True) is not yet implemented; "
            "the unquantized ONNX will be written.",
            UserWarning,
            stacklevel=2,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=_initial_type(),
            target_opset=target_opset,
            options=options,
        )
    onnx.save_model(str(out_path), onnx_model)

    fingerprint = fingerprint_dict(pipeline, opset=target_opset, quantized=False)
    return out_path, fingerprint

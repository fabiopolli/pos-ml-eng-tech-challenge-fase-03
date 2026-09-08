"""ONNX adapter exposing the sklearn ``predict`` / ``predict_proba`` contract.

The adapter normalizes ONNX Runtime outputs to a numpy array of predicted
labels and, when possible, probabilities. sklearn's ``LinearSVC`` has no
``predict_proba``; in that case ``score`` is reported as the margin
(``decision_function`` of the predicted class) and ``score_kind`` is set to
``"decision_function"`` so callers know the value is not a probability.

The adapter keeps ``onnxruntime.InferenceSession`` as a **singleton**
attribute: instantiating a session allocates an arena and several native
handles, and the production API needs to serve thousands of ``/predict``
requests per minute. Re-creating the session every request would defeat the
purpose of the optimization (memory leak, invalidates Δ p95) and was the
#1 bug caught in the post-Fase-2 review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

try:  # pragma: no cover - exercised in optimization environments
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:  # pragma: no cover
    ONNXRUNTIME_AVAILABLE = False


ScoreKind = Literal["predict_proba", "decision_function", "absent"]
_CLASSIFIER_KINDS = ("logreg", "linear_svc", "unknown")
ClassifierKind = Literal["logreg", "linear_svc", "unknown"]


def normalize_classifier_kind(value: str | None) -> ClassifierKind:
    """Map sklearn class names and metadata values onto the adapter kind enum."""

    if not value:
        return "unknown"
    lowered = str(value).strip().lower()
    if lowered in {"logreg", "logisticregression", "logistic_regression"}:
        return "logreg"
    if lowered in {"linear_svc", "linearsvc", "linear_svc_dual", "svc_linear"}:
        return "linear_svc"
    return "unknown"


@dataclass(frozen=True)
class OnnxModelAdapter:
    """Lazy/once-loaded ONNX adapter compatible with the sklearn contract.

    Class attributes:

    * ``classes`` — tuple of int labels (order matches ``model.classes_``).
    * ``classifier_kind`` — one of ``"logreg"``, ``"linear_svc"``,
      ``"unknown"``. Drives the ``predict_proba`` decision.
    * ``input_name`` — cached on first session creation.
    * ``input_type`` — cached on first session creation.
    * ``_session`` — singleton ``ort.InferenceSession``. Created exactly once
      per adapter instance and reused for every ``predict``/``predict_proba``
      /``score_for`` call.
    """

    onnx_path: Path
    classes: tuple[Any, ...]
    classifier_kind: ClassifierKind = "unknown"
    _session_obj: Any = field(default=None, init=False, repr=False, compare=False)
    _input_name: str = field(default="", init=False, repr=False, compare=False)
    _input_type: str = field(default="", init=False, repr=False, compare=False)
    _load_lock_marker: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed; install the [optimization] group")
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")
        kind_options = get_args(ClassifierKind)
        if self.classifier_kind not in kind_options:
            raise ValueError(
                f"classifier_kind must be one of {kind_options}; got {self.classifier_kind!r}"
            )
        if not self.classes:
            raise ValueError("classes must contain at least one label")

    def _ensure_session(self) -> Any:
        """Lazy singleton: create the ``InferenceSession`` once and reuse it.

        Subsequent calls hit the same instance, avoiding the per-request
        memory leak and supporting the Δ p95 latency gate from the Fase 2
        plan.
        """

        if self._session_obj is not None:
            return self._session_obj
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed; install the [optimization] group")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = session.get_inputs()[0].name
        self._input_type = str(session.get_inputs()[0].type or "")
        # ``dataclasses.replace`` is used here to keep ``frozen=True``
        # semantics — direct attribute assignment is rejected by frozen
        # dataclasses.
        object.__setattr__(self, "_session_obj", session)
        object.__setattr__(self, "_load_lock_marker", True)
        return session

    def _prepare(self, texts: list[str]) -> np.ndarray:
        """Adapt ``texts`` to the dtype/shape the ONNX session expects."""

        if "tensor(string)" in self._input_type.lower() or not self._input_type:
            return np.array(texts, dtype=object).reshape(-1, 1)
        return np.array(texts).reshape(-1, 1)

    def _resolve_label(self, raw_label: Any) -> int | None:
        """Map the integer/raw label emitted by ONNX onto ``self.classes``."""

        try:
            return int(raw_label)
        except (TypeError, ValueError):
            return None

    def _label_from_index(self, predicted_index: int) -> int | None:
        if 0 <= predicted_index < len(self.classes):
            return int(self.classes[predicted_index])
        return None

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return the predicted label for each input text."""

        if not texts:
            return np.array([], dtype=int)
        session = self._ensure_session()
        prepared = self._prepare(texts)
        outputs = session.run(None, {self._input_name: prepared})
        label_output = np.asarray(outputs[0]).reshape(-1)
        # ``outputs[0]`` after ``zipmap=False`` carries either the integer
        # label or the index of the predicted class. We normalise both
        # forms to the canonical label tuple member.
        resolved: list[int | None] = []
        for value in label_output:
            label = self._resolve_label(value)
            if label is None:
                # ``zipmap=False`` declared no integer label: fall back to
                # argmax over probabilities when available.
                if len(outputs) >= 2:
                    score = np.asarray(outputs[1])
                    if score.ndim == 2:
                        label = int(np.argmax(score[0]))
                else:
                    label = int(value)
            candidate = (
                self._label_from_index(label) if label < len(self.classes) and label >= 0 else None
            )
            if candidate is None and label in self.classes:
                candidate = int(label)
            if candidate is None:
                # Out-of-range index would explode with IndexError; use a
                # graceful fallback so the caller can decide.
                candidate = int(
                    np.argmax(np.asarray(outputs[1][0])) if len(outputs) >= 2 else value
                )
            resolved.append(candidate)
        return np.asarray(resolved, dtype=int)

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        """Return calibrated probabilities when the underlying model supports them.

        ``LinearSVC`` has no probability surface; in that case ``None`` is
        returned and callers should fall back to ``decision_function``.
        """

        if not texts:
            return np.array([]).reshape(0, len(self.classes))
        if self.classifier_kind != "logreg":
            return None
        session = self._ensure_session()
        prepared = self._prepare(texts)
        outputs = session.run(None, {self._input_name: prepared})
        if len(outputs) >= 2:
            probabilities = np.asarray(outputs[1])
        else:
            probabilities = np.asarray(outputs[0])
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes):
            return None
        return probabilities

    def score_for(self, texts: list[str], predicted_label: Any) -> tuple[float | None, ScoreKind]:
        """Return a confidence-like value for the predicted label.

        Returns the calibrated probability when ``predict_proba`` is available
        (LogReg) or the per-class decision_function margin (LinearSVC). When
        neither surface is informative, returns ``(None, "absent")``.
        """

        if not texts:
            return None, "absent"
        probabilities = self.predict_proba(texts)
        if probabilities is not None and probabilities.size:
            try:
                index = list(self.classes).index(predicted_label)
            except ValueError:
                return None, "absent"
            return float(probabilities[0][index]), "predict_proba"

        session = self._ensure_session()
        prepared = self._prepare(texts)
        outputs = session.run(None, {self._input_name: prepared})
        logits = np.asarray(outputs[0])
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        if logits.shape[1] != len(self.classes):
            return None, "absent"
        try:
            index = list(self.classes).index(predicted_label)
        except ValueError:
            return None, "absent"
        return float(logits[0][index]), "decision_function"

    def close(self) -> None:
        """Release the cached ``InferenceSession`` and reset cached metadata."""

        if self._session_obj is not None:
            session = self._session_obj
            object.__setattr__(self, "_session_obj", None)
            object.__setattr__(self, "_input_name", "")
            object.__setattr__(self, "_input_type", "")
            object.__setattr__(self, "_load_lock_marker", False)
            del session

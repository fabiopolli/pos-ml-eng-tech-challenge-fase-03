"""ONNX adapter exposing the sklearn ``predict`` / ``predict_proba`` contract.

The adapter normalizes ONNX Runtime outputs to a numpy array of predicted
labels and, when possible, probabilities. sklearn's ``LinearSVC`` has no
``predict_proba``; in that case ``score`` is reported as the margin
(``decision_function`` of the predicted class) and ``score_kind`` is set to
``"decision_function"`` so callers know the value is not a probability.

Loading is lazy because instantiating ``onnxruntime.InferenceSession`` involves
disk IO and CPU work best deferred to the first request after startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:  # pragma: no cover - exercised in optimization environments
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:  # pragma: no cover
    ONNXRUNTIME_AVAILABLE = False


ScoreKind = Literal["predict_proba", "decision_function", "absent"]


@dataclass(frozen=True)
class OnnxModelAdapter:
    """Lazy-loading ONNX adapter compatible with the sklearn contract."""

    onnx_path: Path
    classes: tuple[Any, ...]
    classifier_kind: Literal["logreg", "linear_svc", "unknown"]

    def _session(self) -> Any:
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed; install the [optimization] group")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        return ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return the predicted class for each input text."""

        session = self._session()
        input_name = session.get_inputs()[0].name
        prepared = _prepare_str_input(texts, session.get_inputs()[0].type)
        outputs = session.run(None, {input_name: prepared})[0]
        labels = np.asarray(outputs).reshape(-1)
        return np.asarray([self.classes[int(index)] for index in labels])

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        """Return calibrated probabilities when the underlying model supports them.

        ``LinearSVC`` has no probability surface; in that case ``None`` is
        returned and callers should fall back to ``decision_function``.
        """

        session = self._session()
        input_name = session.get_inputs()[0].name
        prepared = _prepare_str_input(texts, session.get_inputs()[0].type)
        outputs = session.run(None, {input_name: prepared})
        # ONNX sklearn conversion with ``zipmap=False`` returns a single
        # (n_examples, n_classes) probability matrix in the first output.
        if len(outputs) >= 2:
            probabilities = np.asarray(outputs[1])
        else:
            probabilities = np.asarray(outputs[0])
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes):
            return None
        if self.classifier_kind == "linear_svc":
            # LinearSVC has no probability surface; ONNX may still emit a
            # softmax over decision_function scores but it is not calibrated.
            return None
        return probabilities

    def score_for(self, texts: list[str], predicted_label: Any) -> tuple[float | None, ScoreKind]:
        """Return a confidence-like value for the predicted label.

        Returns the calibrated probability when ``predict_proba`` is available
        (LogReg) or the per-class decision_function margin (LinearSVC). When
        neither surface is informative, returns ``(None, "absent")``.
        """

        probabilities = self.predict_proba(texts)
        if probabilities is not None:
            try:
                index = list(self.classes).index(predicted_label)
            except ValueError:
                return None, "absent"
            return float(probabilities[0][index]), "predict_proba"

        session = self._session()
        input_name = session.get_inputs()[0].name
        prepared = _prepare_str_input(texts, session.get_inputs()[0].type)
        outputs = session.run(None, {input_name: prepared})
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


def _prepare_str_input(texts: list[str], declared_type: str | None) -> np.ndarray:
    """Adapt the input list to the ONNX expected dtype/shape."""

    if declared_type is None or "tensor(string)" in declared_type.lower():
        return np.array(texts, dtype=object).reshape(-1, 1)
    # Fallback: feed as object array; ``onnxruntime`` will downcast.
    return np.array(texts).reshape(-1, 1)

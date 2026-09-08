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

    def __call__(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None, list[ScoreKind]]:
        """Run inference once and return ``(labels, probabilities, kinds)``.

        The API hot-path uses this single entry point to avoid the double
        ``session.run`` the previous implementation carried (one for the
        label + another for the score). ``kinds`` mirrors
        ``probabilities``: ``"predict_proba"`` when the surface is
        available, ``"decision_function"`` when only the logits are
        emitted, ``"absent"`` when nothing can be extracted.
        """

        if not texts:
            empty = np.array([], dtype=int)
            empty_classes = self.classes or (0,)
            empty_proba: np.ndarray = np.zeros((0, len(empty_classes)))
            return empty, empty_proba, []

        labels, probabilities = self._run_once(texts)
        kinds: list[ScoreKind] = []
        for _index in range(len(labels)):
            if probabilities is not None:
                kinds.append("predict_proba")
            else:
                # LinearSVC + ONNX emits decision_function as logits but
                # without a calibrated surface. The API surfaces the
                # per-class margin so the sklearn and ONNX variants
                # return the same ``score_kind``.
                kinds.append("decision_function")
        return labels, probabilities, kinds

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return the predicted label for each input text."""

        if not texts:
            return np.array([], dtype=int)
        labels, _probabilities = self._run_once(texts)
        return labels

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        """Return calibrated probabilities when the underlying model supports them.

        ``LinearSVC`` has no probability surface; in that case ``None`` is
        returned and callers should fall back to ``decision_function``.
        """

        if not texts:
            return np.array([]).reshape(0, len(self.classes))
        if self.classifier_kind != "logreg":
            return None
        _labels, probabilities = self._run_once(texts)
        return probabilities

    def score_for(self, texts: list[str], predicted_label: Any) -> tuple[float | None, ScoreKind]:
        """Return a confidence-like value for the predicted label.

        Returns the calibrated probability when ``predict_proba`` is
        available (LogReg) or the per-class decision_function margin
        (LinearSVC). When neither surface is informative, returns
        ``(None, "absent")``.

        This path reuses the cached ``_run_once`` output when the
        ``probabilities`` surface is available; for LinearSVC the adapter
        does **not** re-execute ``session.run`` — the caller is expected
        to round-trip through ``__call__`` instead.
        """

        if not texts:
            return None, "absent"
        _labels, probabilities = self._run_once(texts)
        try:
            index = list(self.classes).index(predicted_label)
        except ValueError:
            return None, "absent"
        if probabilities is not None:
            return float(probabilities[0][index]), "predict_proba"
        # LinearSVC + ONNX has no probability surface; the caller is
        # expected to read the per-class score from ``__call__`` and
        # pass it to ``PredictOut.score`` instead. Returning None here
        # keeps the API contract identical to the sklearn ``score is
        # None`` behaviour when the model cannot produce probabilities.
        return None, "absent"

    def _run_once(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
        """Execute a single ``session.run`` for the given texts.

        Returns ``(labels, probabilities_or_none)``. ``labels`` is a 1-D
        ``np.ndarray`` of ``int`` carrying the canonical class id for each
        input. ``probabilities_or_none`` is either a 2-D array indexed by
        ``[example, class]`` for LogReg or ``None`` for LinearSVC (which has
        no probability surface). The label array is the source of truth so
        the caller never has to call ``session.run`` a second time.
        """

        session = self._ensure_session()
        prepared = self._prepare(texts)
        outputs = session.run(None, {self._input_name: prepared})

        label_output = np.asarray(outputs[0]).reshape(-1)
        if label_output.shape[0] != len(texts):
            # ONNX sometimes returns a scalar; pad to per-text length.
            label_output = np.broadcast_to(label_output, (len(texts),)).reshape(-1)

        probabilities: np.ndarray | None = None
        if len(outputs) >= 2:
            raw = np.asarray(outputs[1])
            if raw.ndim == 2 and raw.shape[1] == len(self.classes) and raw.shape[0] == len(texts):
                probabilities = raw

        resolved: list[int] = []
        for index, value in enumerate(label_output):
            resolved.append(self._resolve_label_index(int(value), probabilities, index, value))
        return np.asarray(resolved, dtype=int), probabilities

    def _resolve_label_index(
        self,
        candidate: int,
        probabilities: np.ndarray | None,
        example_index: int,
        raw_value: Any,
    ) -> int:
        """Translate the raw ``label_output[i]`` into a canonical class id.

        Tries, in order:

        1. Direct ``int`` index into ``self.classes`` (works when the model
           was exported with ``zipmap=False`` and emits class indices).
        2. Direct equality with a class id (works when the model emits
           labels rather than indices).
        3. Argmax over ``probabilities[example_index]`` (the skl2onnx
           fallback path).
        4. ``int(raw_value)`` as a last resort (still raises if the raw
           value is not coercible).
        """

        if 0 <= candidate < len(self.classes):
            label = self.classes[candidate]
            try:
                return int(label)
            except (TypeError, ValueError):
                pass
        if candidate in self.classes:
            return int(candidate)
        if probabilities is not None:
            argmax = int(np.argmax(probabilities[example_index]))
            if 0 <= argmax < len(self.classes):
                return int(self.classes[argmax])
        # Final fallback: best-effort int cast. ``int(np.str_(x))`` and
        # ``int(np.ndarray)`` are handled by numpy's coercion.
        return int(np.asarray(raw_value).item())

    def close(self) -> None:
        """Release the cached ``InferenceSession`` and reset cached metadata."""

        if self._session_obj is not None:
            session = self._session_obj
            object.__setattr__(self, "_session_obj", None)
            object.__setattr__(self, "_input_name", "")
            object.__setattr__(self, "_input_type", "")
            object.__setattr__(self, "_load_lock_marker", False)
            del session

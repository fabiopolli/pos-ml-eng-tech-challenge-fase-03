"""Language detection for the smoke API.

We rely on ``langid`` (lid.py) — a small, deterministic language
identifier that ships with a prebuilt model and has no network
dependency.

``langid.classify`` returns a tuple ``(iso_code, log_probability)``
where ``log_probability`` is a non-positive float. We normalise it
back to a confidence value in ``[0, 1]`` via ``math.exp(raw_score)``
so the rest of the API can speak in probabilities consistently.

The module is intentionally framework-agnostic so tests can patch
:func:`detect_language` directly without monkeypatching globals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import langid


@dataclass(frozen=True)
class LanguageCheck:
    """Outcome of a single language check."""

    code: str | None  # ISO 639-1 code, or None when the input was rejected pre-check
    score: float | None  # confidence in [0, 1], None for pre-check rejections
    accepted: bool  # True when the language passes the configured policy


class UnsupportedLanguageError(ValueError):
    """Raised when a request payload fails the language policy."""

    def __init__(self, *, code: str | None, score: float | None, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.score = score
        self.reason = reason


def _normalise_score(raw_score: float) -> float:
    """Convert ``langid``'s log-probability into a confidence in ``[0, 1]``.

    ``langid`` returns the natural log of the probability assigned to
    the predicted class. Anything below about ``log(1e-200)`` saturates
    to ``0``; anything at or above ``0`` saturates to ``1``.
    """

    if raw_score >= 0:
        return 1.0
    if raw_score <= -500:
        return 0.0
    return max(0.0, min(1.0, math.exp(raw_score)))


def detect_language(
    text: str, *, min_chars: int, min_score: float, supported: set[str]
) -> LanguageCheck:
    """Detect the language of ``text`` and apply the API policy.

    Parameters
    ----------
    text:
        Already-stripped input string. Caller is expected to have
        validated the empty/blank case separately.
    min_chars:
        Minimum length (in characters) required for a stable detection.
        Inputs shorter than this raise :class:`UnsupportedLanguageError`
        with ``reason="text_too_short_for_language_check"``.
    min_score:
        Lower bound for the detector's confidence score (in ``[0, 1]``).
        Detections below this value are treated as indeterminate and
        rejected with ``reason="indeterminate_language"``.
    supported:
        ISO 639-1 codes accepted by the API. Anything outside the
        set is rejected with ``reason="unsupported_language"``.
    """

    if len(text) < min_chars:
        raise UnsupportedLanguageError(
            code=None,
            score=None,
            reason="text_too_short_for_language_check",
        )

    try:
        iso_code, raw_score = langid.classify(text)
    except (ValueError, IndexError, ZeroDivisionError) as exc:
        # langid can raise on very short / token-free inputs even after the
        # length guard. Treat those as indeterminate so the API never crashes.
        raise UnsupportedLanguageError(
            code=None,
            score=None,
            reason="indeterminate_language",
        ) from exc
    score = _normalise_score(float(raw_score))
    iso_code = iso_code.lower()

    if score < min_score:
        raise UnsupportedLanguageError(
            code=iso_code,
            score=score,
            reason="indeterminate_language",
        )

    if iso_code not in supported:
        raise UnsupportedLanguageError(
            code=iso_code,
            score=score,
            reason="unsupported_language",
        )

    return LanguageCheck(code=iso_code, score=score, accepted=True)

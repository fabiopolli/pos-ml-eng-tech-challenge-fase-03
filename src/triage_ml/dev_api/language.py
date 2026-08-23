"""Language detection for the dev API.

We rely on ``langid`` (lid.py) — a small, deterministic language
identifier that ships with a prebuilt model and has no network
dependency.

The package-level ``langid.classify`` returns an unnormalised score, so
the API uses its own ``LanguageIdentifier`` configured with
``norm_probs=True``. The resulting score is a normalized probability in
``[0, 1]`` (not a calibrated confidence estimate).

The module is intentionally framework-agnostic so tests can patch
:func:`detect_language` directly without monkeypatching globals.
"""

from __future__ import annotations

from dataclasses import dataclass

from langid.langid import LanguageIdentifier, model

LANGUAGE_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)


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
        iso_code, probability = LANGUAGE_IDENTIFIER.classify(text)
    except (ValueError, IndexError, ZeroDivisionError) as exc:
        # langid can raise on very short / token-free inputs even after the
        # length guard. Treat those as indeterminate so the API never crashes.
        raise UnsupportedLanguageError(
            code=None,
            score=None,
            reason="indeterminate_language",
        ) from exc
    score = float(probability)
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

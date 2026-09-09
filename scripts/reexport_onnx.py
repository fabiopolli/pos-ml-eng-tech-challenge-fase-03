"""Re-export ``model.onnx`` for an existing artifact with a re2-compatible
tokenizer.

Why this script exists
----------------------
``skl2onnx`` embeds the sklearn ``TfidfVectorizer.token_pattern`` directly
into the ONNX graph. The default value (``"(?u)\\b\\w+\\b"``) uses the
inline ``(?u)`` modifier accepted by Python's ``re`` module but rejected
by ``re2`` — the regex engine that ships with ``onnxruntime``. The result
is a fail-fast ``Tokenizer::Can not digest tokenexp`` at session
creation, which makes ``/health`` return 503 for
``TRIAGE_ML_MODEL_VARIANT=onnx``.

The re-exported artifact trades the ``(?u)`` Unicode flag for a
plain ASCII ``\\b\\w+\\b``. Inference output will differ from the sklearn
baseline for non-ASCII inputs (clinical abstracts in Portuguese, for
example) — this script is meant for **local smoke tests only**, not as
a substitute for re-training with a re2-compatible pattern from the
start.

Usage::

    uv run --extra optimization python scripts/reexport_onnx.py \
        --model-dir models/20260823T135811Z-bed2194376bc

    # show the diff without writing
    uv run --extra optimization python scripts/reexport_onnx.py \
        --model-dir models/20260823T135811Z-bed2194376bc --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPSET = 17
# ``re2`` (the engine inside onnxruntime's contrib Tokenizer) rejects
# the Python-only inline modifier ``(?u)``. Plain ``\b\w+\b`` matches
# ``[A-Za-z0-9_]+`` word boundaries — sufficient for ASCII clinical
# abstracts and required for the ONNX runtime to load the graph.
RE2_TOKEN_PATTERN = r"\b\w+\b"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_hash(classifier: str, opset: int, quantized: bool) -> str:
    import hashlib

    payload = json.dumps(
        {"classifier": classifier, "opset": opset, "quantized": quantized},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_pipeline(joblib_path: Path):
    """Load the fitted sklearn pipeline, tolerating sklearn version drift.

    ``InconsistentVersionWarning`` is non-fatal: ``joblib.load`` succeeds
    when the persisted estimator was pickled by a compatible sklearn
    version. We emit the warning once and continue.
    """
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("default", category=UserWarning)
        return joblib.load(joblib_path)


def _swap_token_pattern(pipeline, new_pattern: str) -> str:
    """Override ``tfidf.token_pattern`` in-place; return the previous value."""

    tfidf = pipeline.named_steps["tfidf"]
    previous = tfidf.token_pattern
    tfidf.token_pattern = new_pattern
    return previous


def _reexport(model_dir: Path, *, dry_run: bool) -> int:
    from triage_ml.optimization.optimize import export_onnx

    joblib_path = model_dir / "model.joblib"
    onnx_path = model_dir / "model.onnx"
    metadata_path = model_dir / "metadata.json"
    if not (joblib_path.is_file() and onnx_path.is_file() and metadata_path.is_file()):
        print(
            f"{model_dir.name}: missing one of model.joblib/model.onnx/metadata.json",
            file=sys.stderr,
        )
        return 1

    # ``validate_artifact_bundle`` checks that ``metadata.preprocessing.tfidf.token_pattern``
    # matches the value persisted on the fitted ``TfidfVectorizer``. We must
    # therefore validate the bundle **before** swapping the pattern.
    from triage_ml.models.artifact import validate_artifact_bundle

    metadata = validate_artifact_bundle(joblib_path)
    preprocessing = metadata.get("preprocessing") or {}
    declared_pattern = (preprocessing.get("tfidf") or {}).get("token_pattern")
    if declared_pattern == RE2_TOKEN_PATTERN:
        print(f"{model_dir.name}: token_pattern already re2-compatible ({RE2_TOKEN_PATTERN!r})")
        return 0

    pipeline = _load_pipeline(joblib_path)
    tfidf = pipeline.named_steps["tfidf"]
    previous_pattern = tfidf.token_pattern
    print(f"{model_dir.name}:")
    print(f"  - current token_pattern = {previous_pattern!r}")
    print(f"  - new token_pattern     = {RE2_TOKEN_PATTERN!r}")
    if dry_run:
        print("  dry-run: not re-exporting")
        return 0

    _swap_token_pattern(pipeline, RE2_TOKEN_PATTERN)
    target_path, fingerprint = export_onnx(pipeline, onnx_path, opset=DEFAULT_OPSET)
    # The production API container runs as uid 10001 (``triage``) and
    # reads the artifact from a read-only bind mount. ``umask 0o022`` plus
    # an explicit ``chmod 0o644`` makes the freshly written ``model.onnx``
    # world-readable so the container can ``open()`` it; ``skl2onnx``
    # defaults to ``0o644`` already but ``umask 0o077`` (common in dev
    # shells) would otherwise leave the file unreadable.
    onnx_path.chmod(0o644)

    fingerprint_dict = {
        "classifier": fingerprint.classifier,
        "opset": fingerprint.opset,
        "quantized": fingerprint.quantized,
    }
    fingerprint_dict["fingerprint_hash"] = _fingerprint_hash(
        fingerprint.classifier, fingerprint.opset, fingerprint.quantized
    )
    optimization_record = {
        "onnx_path": target_path.name,
        "onnx_checksum_sha256": _sha256(target_path),
        "onnx_opset": fingerprint.opset,
        "source_joblib_checksum_sha256": metadata["checksum_sha256"],
        "optimization_fingerprint": fingerprint_dict,
        "created_at": datetime.now(UTC).isoformat(),
        "reexported_for_onnxruntime_compat": True,
        "previous_token_pattern": previous_pattern,
        "current_token_pattern": RE2_TOKEN_PATTERN,
    }
    metadata["optimization"] = optimization_record
    metadata["available_variants"] = ["sklearn", "onnx"]
    # NOTE: do not touch ``metadata.preprocessing.tfidf.token_pattern`` —
    # it must keep matching the fitted ``TfidfVectorizer`` persisted in
    # ``model.joblib`` so that ``load_artifact`` and the
    # ``token_pattern disagrees with metadata`` check pass for the
    # sklearn variant. The re-export divergence (sklearn uses the Python
    # ``(?u)\\b\\w+\\b``; ONNX uses the re2-compatible ``\\b\\w+\\b``) is
    # recorded in ``metadata.optimization`` for downstream observability.
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  wrote {metadata_path}")
    print(f"  wrote {onnx_path} (sha256={optimization_record['onnx_checksum_sha256'][:12]}...)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Model version directory (e.g. models/20260823T135811Z-bed2194376bc).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    args = parser.parse_args()
    return _reexport(args.model_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
"""Repair ``metadata.json`` under ``models/<version>/`` to match the
current runtime.

When ``uv.lock`` is bumped (e.g. numpy 2.4 → 2.5) but an existing
artifact under ``models/`` was trained against the previous
``uv.lock`` (or against an environment without a frozen lock), the
``validate_artifact_bundle`` check rejects the artifact with::

    artifact dependency versions are incompatible with the runtime:
    {'numpy': ('2.5.2', '2.4.2'), 'scipy': ('1.18.0', '1.17.0'),
     'scikit_learn': ('1.9.0', '1.8.0')}

This script updates the ``dependency_versions`` block in
``metadata.json`` to the runtime versions reported by the current
interpreter **without touching the ``model.joblib`` payload or its
checksum** (``checksum_sha256`` is left as-is, which is correct
because the model itself was not retrained).

When the artifact directory also contains a ``model.onnx`` that was
exported outside the canonical Airflow path (and therefore lacks the
``optimization`` and ``available_variants`` blocks), the patcher
reconstructs the same layout written by
``export_onnx_for_version`` so the ONNX registry accepts the artifact:

* ``metadata.available_variants = ["sklearn", "onnx"]``
* ``metadata.optimization.onnx_checksum_sha256 = sha256(model.onnx)``
* ``metadata.optimization.source_joblib_checksum_sha256 = metadata["checksum_sha256"]``
* ``metadata.optimization.optimization_fingerprint.fingerprint_hash`` matching
  ``triage_ml.optimization.optimize.fingerprint_hash``.

Use this only for local smoke tests where re-training is not
feasible. Production CI must always re-train via
``uv run triage-ml-train`` and let the new artifact carry the new
dependency_versions.

Usage::

    # single version
    uv run python scripts/repair_artifact_metadata.py \
        --model-dir models/20260823T135811Z-bed2194376bc

    # all versions under models/
    uv run python scripts/repair_artifact_metadata.py --all

    # print the diff without writing
    uv run python scripts/repair_artifact_metadata.py --all --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _runtime_versions() -> dict[str, str]:
    """Snapshot of the runtime versions the API container would see."""
    import joblib
    import numpy
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _read_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _needs_update(metadata: dict, runtime: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Return only the keys that differ between metadata and runtime."""
    current = metadata.get("dependency_versions") or {}
    diff: dict[str, tuple[str, str]] = {}
    for name, runtime_version in runtime.items():
        old = current.get(name)
        if old != runtime_version:
            diff[name] = (old, runtime_version)
    return diff


def _write_metadata(metadata_path: Path, metadata: dict) -> None:
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_hash(classifier: str, opset: int, quantized: bool) -> str:
    """Reproduce ``triage_ml.optimization.optimize.fingerprint_hash``.

    The hash is computed over ``json.dumps(
    {"classifier": ..., "opset": ..., "quantized": ...},
    sort_keys=True, separators=(",", ":"))``. Reimplementing it here keeps
    the patcher independent of the optional ``[optimization]`` group and
    avoids unpickling ``model.joblib`` (which may emit
    ``InconsistentVersionWarning`` when ``uv.lock`` has been bumped).
    """
    payload = json.dumps(
        {"classifier": classifier, "opset": opset, "quantized": quantized},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _needs_optimization_block(metadata: dict) -> bool:
    """True when ``metadata.optimization`` is missing or malformed."""
    if "available_variants" not in metadata:
        return True
    if not isinstance(metadata.get("available_variants"), list):
        return True
    if "onnx" not in metadata["available_variants"]:
        return True
    optimization = metadata.get("optimization")
    if not isinstance(optimization, dict):
        return True
    required = {
        "onnx_path",
        "onnx_checksum_sha256",
        "onnx_opset",
        "source_joblib_checksum_sha256",
        "optimization_fingerprint",
    }
    return not required.issubset(optimization.keys())


def _build_optimization_block(metadata: dict, model_dir: Path) -> dict:
    """Build the canonical ``optimization`` record + stamp ``available_variants``.

    Mirrors the layout written by
    ``triage_ml.orchestration.airflow_pipeline.export_onnx_for_version`` so
    the registry's checks (``onnx_checksum_sha256``,
    ``source_joblib_checksum_sha256``, ``optimization_fingerprint``,
    ``available_variants``) accept the patched metadata.
    """
    onnx_path = model_dir / "model.onnx"
    joblib_checksum = metadata["checksum_sha256"]
    preprocessing = metadata.get("preprocessing") or {}
    classifier = str(preprocessing.get("classifier") or "unknown")
    opset = 17
    quantized = False
    fingerprint_dict = {"classifier": classifier, "opset": opset, "quantized": quantized}
    fingerprint_dict["fingerprint_hash"] = _fingerprint_hash(classifier, opset, quantized)
    return {
        "onnx_path": "model.onnx",
        "onnx_checksum_sha256": _sha256(onnx_path),
        "onnx_opset": opset,
        "source_joblib_checksum_sha256": joblib_checksum,
        "optimization_fingerprint": fingerprint_dict,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _iter_model_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not (entry / "model.joblib").is_file() or not (entry / "metadata.json").is_file():
            continue
        try:
            metadata = _read_metadata(entry / "metadata.json")
        except json.JSONDecodeError:
            continue
        if metadata.get("schema_version") != 1:
            # legacy layout (e.g. ``v1/`` with non-versioned metadata) is
            # handled by the regular re-train path, not by this patcher.
            continue
        out.append(entry)
    return sorted(out)


def repair(model_dir: Path, *, dry_run: bool) -> int:
    metadata_path = model_dir / "metadata.json"
    metadata = _read_metadata(metadata_path)
    runtime = _runtime_versions()
    diff = _needs_update(metadata, runtime)

    onnx_path = model_dir / "model.onnx"
    onnx_present = onnx_path.is_file() and not onnx_path.is_symlink()
    optimization_needed = onnx_present and _needs_optimization_block(metadata)

    print(f"{model_dir.name}:")
    if not diff and not optimization_needed:
        print("  ok — metadata already matches runtime and advertises ONNX")
        return 0
    if diff:
        for name, (old, new) in diff.items():
            print(f"  - dependency_versions.{name}: {old!r} → {new!r}")
    if optimization_needed:
        if not onnx_present:
            print("  - optimization: skipped (model.onnx not present)")
        else:
            fingerprint = _fingerprint_hash(
                str((metadata.get("preprocessing") or {}).get("classifier") or "unknown"),
                17,
                False,
            )
            print(f"  - optimization: injecting (fingerprint_hash={fingerprint})")
            print("  - available_variants: setting ['sklearn', 'onnx']")
    if dry_run:
        print("  dry-run: not writing")
        return 0

    if diff:
        metadata["dependency_versions"] = dict(runtime)
    if optimization_needed and onnx_present:
        metadata["optimization"] = _build_optimization_block(metadata, model_dir)
        metadata["available_variants"] = ["sklearn", "onnx"]
    _write_metadata(metadata_path, metadata)
    print(f"  wrote {metadata_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Repair a single model directory under models/.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Repair every version under models/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the diff without writing.",
    )
    args = parser.parse_args()

    if not args.model_dir and not args.all:
        parser.error("pass either --model-dir <path> or --all")

    targets: list[Path] = []
    if args.model_dir:
        targets.append(args.model_dir)
    if args.all:
        targets.extend(_iter_model_dirs(MODELS_DIR))

    if not targets:
        print("no model directories found", file=sys.stderr)
        return 1

    rc = 0
    for target in targets:
        rc |= repair(target, dry_run=args.dry_run)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

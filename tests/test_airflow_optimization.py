"""Tests for the Fase 2 optimization helpers in ``triage_ml.orchestration.airflow_pipeline``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage_ml.orchestration import airflow_pipeline


def _make_minimal_logreg_bundle(tmp_path: Path) -> Path:
    """Skip the canonical trainer and emit a minimal but loaded artifact.

    The canonical trainer enforces the 2_000-row contract (ADR 0003); tests
    for the optimization helpers do not need a real training run, so we
    synthesise a small but loaded ``model.joblib`` + ``metadata.json`` pair
    directly.
    """

    version_dir = tmp_path / "models" / "20260101T000000Z-0123456789ab"
    version_dir.mkdir(parents=True)
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=2)),
            ("clf", LogisticRegression(max_iter=200, random_state=7)),
        ]
    )
    pipeline.fit(
        [f"unique medical abstract number {i:04d} about cardiology or oncology" for i in range(80)],
        [1] * 30 + [2] * 30 + [3] * 20,
    )
    joblib.dump(pipeline, version_dir / "model.joblib")
    metadata = {
        "schema_version": 1,
        "model_version": "20260101T000000Z-0123456789ab",
        "model_name": "triage_test",
        "task_type": "multiclass_text_classification",
        "language": "en",
        "classes": [1, 2, 3],
        "label_mapping": {"1": "neoplasms", "2": "cardio", "3": "nervous"},
        "random_state": 7,
        "n_train": 64,
        "n_test": 16,
        "metrics": {
            "accuracy": 0.9,
            "balanced_accuracy": 0.9,
            "macro_f1": 0.9,
            "weighted_f1": 0.9,
            "per_class": {
                "1": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 6},
                "2": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 6},
                "3": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 4},
            },
        },
        "preprocessing": {
            "vectorizer": "tfidf",
            "tfidf": {},
            "classifier": "logreg",
            "classifier_params": {},
        },
        "selection": {
            "metric": "macro_f1",
            "cv": "StratifiedKFold",
            "folds": 2,
            "candidates": {
                "logreg": {
                    "fold_macro_f1": [0.9, 0.9],
                    "mean_macro_f1": 0.9,
                    "std_macro_f1": 0.0,
                },
                "linear_svc": {
                    "fold_macro_f1": [0.9, 0.9],
                    "mean_macro_f1": 0.9,
                    "std_macro_f1": 0.0,
                },
            },
            "best_classifier": "logreg",
            "selected_classifier": "logreg",
            "selection_policy": "explicit_override",
            "test_set_used_for_selection": False,
        },
        "dependency_versions": {
            "python": "3.12.11",
            "numpy": "2.4.2",
            "scipy": "1.17.0",
            "scikit_learn": "1.8.0",
            "joblib": "1.5.3",
        },
        "git_commit": "0" * 40,
        "git_dirty": False,
        "fingerprints": {
            "raw_csv_sha256": "a" * 64,
            "prepared_dataset_sha256": "b" * 64,
            "train_split_sha256": "c" * 64,
            "test_split_sha256": "d" * 64,
            "config_sha256": "e" * 64,
        },
        "checksum_sha256": "f" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return version_dir


def test_benchmark_for_version_records_sklearn_run(tmp_path: Path) -> None:
    """`benchmark_for_version` should always produce both benchmark JSONs."""

    artifact_dir = _make_minimal_logreg_bundle(tmp_path)
    cwd = tmp_path / "run"
    cwd.mkdir()

    # Pre-create the aggregate directory in the cwd-then-call pattern: the
    # helper returns the absolute paths it just wrote, and we read them
    # back directly without chdir-ing back.
    aggregate_dir = cwd / "reports" / "benchmarks"
    aggregate_dir.mkdir(parents=True)
    (aggregate_dir / "benchmark.json").write_text("{}", encoding="utf-8")

    original_cwd = Path.cwd()
    os.chdir(cwd)
    try:
        result = airflow_pipeline.benchmark_for_version(artifact_dir, benchmark_input_size=8)
    finally:
        os.chdir(original_cwd)

    assert result["sklearn"]["variant"] == "sklearn"
    assert result["onnx"] is None  # no model.onnx next to model.joblib
    assert Path(result["benchmark_paths"]["sibling"]).read_text(encoding="utf-8")
    assert Path(result["benchmark_paths"]["aggregate"]).read_text(encoding="utf-8")


def test_build_optimization_manifest_emits_default_variants() -> None:
    payload = airflow_pipeline.build_optimization_manifest(
        "any", available_variants=("sklearn", "onnx")
    )
    assert payload["available_variants"] == ["sklearn", "onnx"]
    assert "build_utc" in payload


def test_export_onnx_for_version_requires_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the [optimization] extras the helper must raise a helpful error."""

    artifact_dir = _make_minimal_logreg_bundle(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated absence of [optimization] group")

    monkeypatch.setattr("triage_ml.optimization.optimize.export_onnx", boom, raising=True)
    with pytest.raises(RuntimeError):
        airflow_pipeline.export_onnx_for_version(artifact_dir)


def test_export_onnx_for_version_emits_metadata_when_extras_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = _make_minimal_logreg_bundle(tmp_path)

    def fake_export(pipeline, out_path, *, opset, **kwargs):
        Path(out_path).write_bytes(b"onnx-bytes")
        from triage_ml.optimization.optimize import OptimizationFingerprint

        return Path(out_path), OptimizationFingerprint(
            classifier="LogisticRegression", opset=opset, quantized=False
        )

    monkeypatch.setattr("triage_ml.optimization.optimize.export_onnx", fake_export)

    payload = airflow_pipeline.export_onnx_for_version(artifact_dir, opset=17)
    assert payload["optimization"]["onnx_checksum_sha256"]
    assert payload["optimization"]["onnx_opset"] == 17
    assert (artifact_dir / "model.onnx").is_file()

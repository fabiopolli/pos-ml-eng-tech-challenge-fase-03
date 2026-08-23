"""Tests for immutable, validated model artifacts."""

from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage_ml.models.artifact import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    ArtifactPaths,
    build_metadata,
    file_sha256,
    load_artifact,
    read_classes,
    read_metadata,
    validate_metadata,
    verify_artifact_integrity,
    write_classes,
    write_metadata,
)

VERSION = "20260823T120000Z-0123456789ab"
DIGEST = "a" * 64


@pytest.fixture()
def tiny_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipeline.fit(["liver tumor", "heart attack", "liver metastases"], [1, 4, 1])
    return pipeline


def _metadata(joblib_path: Path, pipeline: Pipeline) -> dict[str, object]:
    return build_metadata(
        model_version=VERSION,
        model_name="tiny",
        task_type="multiclass_text_classification",
        language="en",
        classes=list(pipeline.classes_),
        label_mapping={"1": "neoplasms", "4": "cardiovascular diseases"},
        random_state=42,
        n_train=3,
        n_test=2,
        metrics={
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
            "per_class": {
                label: {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "support": 1,
                }
                for label in ("1", "4")
            },
        },
        preprocessing={
            "vectorizer": "tfidf",
            "tfidf": {},
            "classifier": "logreg",
            "classifier_params": {},
        },
        selection={
            "metric": "macro_f1",
            "folds": 2,
            "candidates": {
                name: {
                    "fold_macro_f1": [1.0, 1.0],
                    "mean_macro_f1": 1.0,
                    "std_macro_f1": 0.0,
                }
                for name in ("logreg", "linear_svc")
            },
            "best_classifier": "logreg",
            "selected_classifier": "logreg",
            "selection_policy": "highest_mean_macro_f1",
            "test_set_used_for_selection": False,
        },
        dependency_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        git_commit="0" * 40,
        git_dirty=False,
        fingerprints={
            "raw_csv_sha256": DIGEST,
            "prepared_dataset_sha256": DIGEST,
            "train_split_sha256": DIGEST,
            "test_split_sha256": DIGEST,
            "config_sha256": DIGEST,
        },
        joblib_path=joblib_path,
    )


def _write_artifact(tmp_path: Path, pipeline: Pipeline) -> ArtifactPaths:
    paths = ArtifactPaths.for_version(tmp_path, VERSION)
    paths.ensure()
    joblib.dump(pipeline, paths.joblib)
    write_classes(paths.classes, pipeline.classes_)
    write_metadata(paths.metadata, _metadata(paths.joblib, pipeline))
    return paths


def test_artifact_paths_are_immutable(tmp_path: Path) -> None:
    paths = ArtifactPaths.for_version(tmp_path, VERSION)
    paths.ensure()
    with pytest.raises(FileExistsError):
        paths.ensure()


def test_write_and_read_classes_roundtrip(tmp_path: Path, tiny_pipeline: Pipeline) -> None:
    path = tmp_path / "classes.json"
    assert write_classes(path, tiny_pipeline.classes_) == [1, 4]
    assert read_classes(path) == [1, 4]


def test_metadata_roundtrip_and_checksum(tmp_path: Path, tiny_pipeline: Pipeline) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    metadata = read_metadata(paths.metadata)
    validate_metadata(metadata)
    assert metadata["checksum_sha256"] == file_sha256(paths.joblib)
    assert metadata["label_mapping"] == {"1": "neoplasms", "4": "cardiovascular diseases"}


def test_validate_metadata_rejects_incomplete_or_invalid_schema(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    metadata = read_metadata(paths.metadata)
    del metadata["fingerprints"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_metadata(metadata)

    metadata = read_metadata(paths.metadata)
    metadata["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported"):
        validate_metadata(metadata)


def test_load_artifact_roundtrip_validates_model_classes(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    loaded, metadata = load_artifact(paths.joblib)
    assert list(loaded.classes_) == metadata["classes"] == [1, 4]


def test_checksum_is_checked_before_joblib_deserialization(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    paths.joblib.write_bytes(b"replaced")
    with patch("joblib.load") as mocked_load:
        with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
            load_artifact(paths.joblib)
    mocked_load.assert_not_called()


def test_load_artifact_rejects_classes_file_mismatch(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    write_classes(paths.classes, [1, 2])
    with pytest.raises(ArtifactCompatibilityError, match="classes.json"):
        load_artifact(paths.joblib)


def test_load_artifact_rejects_version_directory_mismatch(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    wrong_dir = tmp_path / "20260823T120001Z-0123456789ab"
    paths.version_dir.rename(wrong_dir)
    with pytest.raises(ArtifactCompatibilityError, match="directory"):
        load_artifact(wrong_dir / "model.joblib")


def test_load_artifact_rejects_declared_parameter_mismatch(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    metadata = read_metadata(paths.metadata)
    metadata["preprocessing"]["tfidf"] = {"min_df": 999}
    write_metadata(paths.metadata, metadata)
    with pytest.raises(ArtifactCompatibilityError, match="parameter 'min_df'"):
        load_artifact(paths.joblib)


def test_load_artifact_rejects_symlinked_model(tmp_path: Path, tiny_pipeline: Pipeline) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    real_joblib = paths.joblib.with_name("model-real.joblib")
    paths.joblib.rename(real_joblib)
    paths.joblib.symlink_to(real_joblib.name)
    with pytest.raises(ArtifactCompatibilityError, match="symlinks"):
        load_artifact(paths.joblib)


def test_verify_artifact_integrity_requires_valid_checksum(tmp_path: Path) -> None:
    joblib_path = tmp_path / "model.joblib"
    joblib_path.write_bytes(b"x")
    with pytest.raises(ArtifactIntegrityError, match="invalid checksum"):
        verify_artifact_integrity(joblib_path=joblib_path, metadata={})


def test_metadata_rejects_impossible_version_timestamp(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    metadata = read_metadata(paths.metadata)
    metadata["model_version"] = "20261340T256199Z-0123456789ab"

    with pytest.raises(ValueError, match="timestamp"):
        validate_metadata(metadata)


def test_dependency_version_is_checked_before_deserialization(
    tmp_path: Path, tiny_pipeline: Pipeline
) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    metadata = read_metadata(paths.metadata)
    metadata["dependency_versions"]["scikit_learn"] = "0.0.0"
    write_metadata(paths.metadata, metadata)

    with patch("joblib.load") as mocked_load:
        with pytest.raises(ArtifactCompatibilityError, match="dependency versions"):
            load_artifact(paths.joblib)
    mocked_load.assert_not_called()


def test_load_artifact_requires_classes_manifest(tmp_path: Path, tiny_pipeline: Pipeline) -> None:
    paths = _write_artifact(tmp_path, tiny_pipeline)
    paths.classes.unlink()

    with pytest.raises(FileNotFoundError, match="classes.json"):
        load_artifact(paths.joblib)

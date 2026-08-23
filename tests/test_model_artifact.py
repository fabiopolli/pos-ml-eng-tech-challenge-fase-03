"""Tests for the artifact reader/writer helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage_ml.models.artifact import (
    REQUIRED_METADATA_KEYS,
    ArtifactPaths,
    build_metadata,
    file_sha256,
    read_classes,
    read_metadata,
    validate_metadata,
    write_classes,
    write_metadata,
)


@pytest.fixture()
def tiny_pipeline() -> Pipeline:
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 1), min_df=1, token_pattern=r"(?u)\b\w+\b"),
            ),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(
        ["liver tumor", "heart attack", "liver metastases"],
        [1, 4, 1],
    )
    return pipe


def test_artifact_paths_layout(tmp_path: Path) -> None:
    paths = ArtifactPaths.for_version(tmp_path, "v1")
    assert paths.version_dir == tmp_path / "v1"
    assert paths.joblib.name == "model.joblib"
    assert paths.classes.name == "classes.json"
    assert paths.metadata.name == "metadata.json"


def test_artifact_paths_ensure_creates_directory(tmp_path: Path) -> None:
    paths = ArtifactPaths.for_version(tmp_path, "v2")
    paths.ensure()
    assert paths.version_dir.exists()


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    assert file_sha256(f) == file_sha256(f)


def test_write_and_read_classes_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "classes.json"
    out = write_classes(path, [1, 4, 2, 3])
    assert out == [1, 4, 2, 3]
    assert read_classes(path) == [1, 4, 2, 3]


def test_write_classes_coerces_numpy_int(tmp_path: Path, tiny_pipeline: Pipeline) -> None:
    path = tmp_path / "classes.json"
    out = write_classes(path, tiny_pipeline.classes_)
    # Persisted file must contain JSON-native integers, not numpy.int64
    raw = path.read_text(encoding="utf-8")
    json.loads(raw)
    assert all(isinstance(c, int) for c in out)


def test_metadata_roundtrip(tmp_path: Path, tiny_pipeline: Pipeline) -> None:
    paths = ArtifactPaths.for_version(tmp_path, "v1")
    paths.ensure()
    joblib.dump(tiny_pipeline, paths.joblib)
    metadata = build_metadata(
        model_version="v1",
        model_name="tiny",
        classes=list(tiny_pipeline.classes_),
        random_state=42,
        n_train=3,
        n_test=1,
        metrics={"accuracy": 1.0},
        preprocessing={"vectorizer": "tfidf"},
        joblib_path=paths.joblib,
        sklearn_version="1.5.0",
        numpy_version="1.26.0",
        python_version="3.12",
    )
    write_metadata(paths.metadata, metadata)
    loaded = read_metadata(paths.metadata)
    assert loaded == metadata
    assert loaded["checksum_sha256"] == file_sha256(paths.joblib)


def test_validate_metadata_requires_all_keys() -> None:
    metadata = {"model_version": "v1"}
    with pytest.raises(ValueError, match="missing required keys"):
        validate_metadata(metadata)


def test_validate_metadata_accepts_complete_metadata(tmp_path: Path) -> None:
    paths = ArtifactPaths.for_version(tmp_path, "v1")
    paths.ensure()
    (paths.joblib).write_bytes(b"x")
    metadata = {key: "placeholder" for key in REQUIRED_METADATA_KEYS}
    metadata["checksum_sha256"] = file_sha256(paths.joblib)
    validate_metadata(metadata)  # should not raise


def test_verify_artifact_integrity_passes(tmp_path: Path) -> None:
    from triage_ml.models.artifact import verify_artifact_integrity

    paths = ArtifactPaths.for_version(tmp_path, "v1")
    paths.ensure()
    paths.joblib.write_bytes(b"joblib-binary-blob")
    metadata = {key: "x" for key in REQUIRED_METADATA_KEYS}
    metadata["checksum_sha256"] = file_sha256(paths.joblib)
    verify_artifact_integrity(joblib_path=paths.joblib, metadata=metadata)


def test_verify_artifact_integrity_detects_swap(tmp_path: Path) -> None:
    from triage_ml.models.artifact import ArtifactIntegrityError, verify_artifact_integrity

    paths = ArtifactPaths.for_version(tmp_path, "v1")
    paths.ensure()
    paths.joblib.write_bytes(b"original")
    metadata = {key: "x" for key in REQUIRED_METADATA_KEYS}
    metadata["checksum_sha256"] = file_sha256(paths.joblib)
    # Silent model swap: overwrite the joblib after metadata was written.
    paths.joblib.write_bytes(b"replaced-by-an-attacker")
    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        verify_artifact_integrity(joblib_path=paths.joblib, metadata=metadata)


def test_verify_artifact_integrity_requires_checksum(tmp_path: Path) -> None:
    from triage_ml.models.artifact import ArtifactIntegrityError, verify_artifact_integrity

    paths = ArtifactPaths.for_version(tmp_path, "v1")
    paths.ensure()
    paths.joblib.write_bytes(b"x")
    metadata = {"model_version": "v1"}
    with pytest.raises(ArtifactIntegrityError, match="missing checksum"):
        verify_artifact_integrity(joblib_path=paths.joblib, metadata=metadata)

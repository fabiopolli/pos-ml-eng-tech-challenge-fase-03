"""Integration tests for selection and training provenance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from triage_ml.data.prepare import PreparationReport
from triage_ml.models.artifact import load_artifact
from triage_ml.models.train import run_training


def test_run_training_records_selection_metrics_and_fingerprints(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = pd.DataFrame(
        {
            "text": [
                f"category{label} shared medical words unique{index}"
                for label in range(1, 6)
                for index in range(50)
            ],
            "target": [label for label in range(1, 6) for _ in range(50)],
        }
    )
    report = PreparationReport(
        input_rows=len(canonical),
        missing_or_empty_rows=0,
        conflicting_texts=0,
        conflicting_rows=0,
        duplicate_rows=0,
        eligible_rows=len(canonical),
        output_rows=len(canonical),
    )

    def fake_prepare(*_args, **_kwargs):
        return canonical, report

    monkeypatch.setattr("triage_ml.models.train.prepare_dataset", fake_prepare)
    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text("medical_abstract,condition_label\nplaceholder,1\n", encoding="utf-8")
    config = {
        "random_state": 42,
        "sample_size": 2000,
        "test_size": 0.2,
        "cv_folds": 2,
        "language": "en",
        "task_type": "multiclass_text_classification",
        "label_mapping": {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
        },
        "tfidf": {"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0},
        "logreg": {"max_iter": 200, "class_weight": "balanced", "C": 1.0},
        "linear_svc": {"class_weight": "balanced", "C": 1.0},
        "model_name": "tiny",
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_training(
        raw_csv_path=raw_csv,
        out_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
        config_path=config_path,
    )
    metadata = summary["metadata"]
    assert metadata["selection"]["test_set_used_for_selection"] is False
    assert set(metadata["selection"]["candidates"]) == {"logreg", "linear_svc"}
    assert "balanced_accuracy" in metadata["metrics"]
    assert set(metadata["fingerprints"]) == {
        "raw_csv_sha256",
        "prepared_dataset_sha256",
        "train_split_sha256",
        "test_split_sha256",
        "config_sha256",
    }
    pipeline, loaded_metadata = load_artifact(summary["paths"]["joblib"])
    assert list(pipeline.classes_) == loaded_metadata["classes"] == [1, 2, 3, 4, 5]
    assert loaded_metadata["preprocessing"]["tfidf"]["sublinear_tf"] is True
    assert loaded_metadata["preprocessing"]["classifier_params"]["random_state"] == 42
    assert Path(summary["paths"]["summary"]).is_file()
    assert Path(summary["paths"]["confusion_matrix"]).parent.name == summary["model_version"]
    assert Path(summary["paths"]["top_features"]).is_file()
    version_dirs = list((tmp_path / "models").iterdir())
    assert version_dirs == [Path(summary["paths"]["joblib"]).parent]

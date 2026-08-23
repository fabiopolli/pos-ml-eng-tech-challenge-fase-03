"""Train, evaluate and serialize the baseline text classifier.

The public entry point is :func:`run_training`, which loads the raw
``medical_tc_train.csv`` dataset, applies the project preparation
contract (``triage_ml.data.prepare``), fits a TF-IDF + linear classifier
pipeline and serializes the artifact under ``models/<version>/``.

The function also generates the figures that back the README and the
optimization report: confusion matrix and per-class top features.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from triage_ml.data.prepare import prepare_dataset, split_dataset
from triage_ml.models.artifact import (
    ArtifactPaths,
    build_metadata,
    validate_metadata,
    write_classes,
    write_metadata,
)
from triage_ml.models.pipeline import VALID_CLASSIFIERS, build_pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training.yaml"
DEFAULT_RAW_CSV = REPO_ROOT / "data" / "medical_tc_train.csv"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "figures"


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML configuration that drives ``run_training``."""

    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping")
    return config


def _numpy_version() -> str:
    return np.__version__


def _coerce_tfidf(tfidf: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert YAML-friendly values to types accepted by sklearn."""

    if not tfidf:
        return tfidf
    coerced = dict(tfidf)
    ngram = coerced.get("ngram_range")
    if isinstance(ngram, list):
        coerced["ngram_range"] = tuple(ngram)
    return coerced


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted F1 and a per-class report."""

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    accuracy = float(report["accuracy"])
    return {
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            str(int(label)): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in labels
        },
    }


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[int],
    out_path: Path,
) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_top_features(
    pipeline: Pipeline,
    labels: list[int],
    out_path: Path,
    top_n: int = 12,
) -> Path | None:
    """Plot the top-N coefficients per class for a linear classifier."""

    clf = pipeline.named_steps["clf"]
    tfidf: Any = pipeline.named_steps["tfidf"]
    feature_names = tfidf.get_feature_names_out()
    classes = list(getattr(clf, "classes_", labels))

    if not hasattr(clf, "coef_"):
        return None

    fig, axes = plt.subplots(
        nrows=len(classes), ncols=1, figsize=(8, 1.6 * len(classes)), sharex=True
    )
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes, strict=False):
        idx = classes.index(cls)
        coefs = clf.coef_[idx]
        top = np.argsort(coefs)[-top_n:][::-1]
        tokens = [feature_names[i] for i in top]
        scores = coefs[top]
        ax.barh(range(len(tokens))[::-1], scores[::-1], color="steelblue")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens[::-1])
        ax.set_title(f"Top features for class {cls}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_training(
    *,
    raw_csv_path: str | Path = DEFAULT_RAW_CSV,
    out_dir: str | Path = DEFAULT_MODELS_DIR,
    figures_dir: str | Path = DEFAULT_REPORTS_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
    classifier: str | None = None,
    sample_size: int | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline and return a summary dictionary."""

    config = load_config(Path(config_path))
    classifier_name = classifier or config["classifier"]
    if classifier_name not in VALID_CLASSIFIERS:
        raise ValueError(f"Unsupported classifier {classifier_name!r}; valid: {VALID_CLASSIFIERS}")
    sample = int(sample_size or config["sample_size"])
    test_frac = float(test_size or config["test_size"])
    seed = int(random_state or config["random_state"])
    model_version = str(config.get("model_version", "v1"))
    model_name = str(config.get("model_name", "triage_ml_tfidf_logreg"))

    raw = pd.read_csv(raw_csv_path)
    canonical, report = prepare_dataset(raw, sample_size=sample, random_state=seed)
    train_df, test_df = split_dataset(canonical, test_size=test_frac, random_state=seed)
    labels = sorted(canonical["target"].unique().tolist())

    pipeline = build_pipeline(
        classifier_name,
        tfidf=_coerce_tfidf(config.get("tfidf")),
        classifier_kwargs=config.get(classifier_name),
    )
    pipeline.fit(train_df["text"], train_df["target"])

    y_pred = pipeline.predict(test_df["text"])
    metrics = compute_metrics(test_df["target"], y_pred, labels)

    paths = ArtifactPaths.for_version(out_dir, model_version)
    paths.ensure()
    joblib.dump(pipeline, paths.joblib)
    persisted_classes = write_classes(paths.classes, pipeline.classes_)
    metadata = build_metadata(
        model_version=model_version,
        model_name=model_name,
        classes=[int(c) for c in persisted_classes],
        random_state=seed,
        n_train=len(train_df),
        n_test=len(test_df),
        metrics=metrics,
        preprocessing={
            "vectorizer": "tfidf",
            "tfidf": config.get("tfidf", {}),
            "classifier": classifier_name,
            "classifier_params": config.get(classifier_name, {}),
        },
        joblib_path=paths.joblib,
        sklearn_version=sklearn.__version__,
        numpy_version=_numpy_version(),
        python_version=platform.python_version(),
    )
    write_metadata(paths.metadata, metadata)
    validate_metadata(metadata)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plot_confusion_matrix(
        test_df["target"], y_pred, labels, figures_dir / "08_confusion_matrix_lr.png"
    )
    top_path = plot_top_features(pipeline, labels, figures_dir / "08_top_features_lr.png")

    return {
        "model_version": model_version,
        "model_name": model_name,
        "classifier": classifier_name,
        "paths": {
            "joblib": str(paths.joblib),
            "classes": str(paths.classes),
            "metadata": str(paths.metadata),
            "confusion_matrix": str(cm_path),
            "top_features": str(top_path) if top_path else None,
        },
        "metrics": metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "preparation_report": {
            "input_rows": report.input_rows,
            "missing_or_empty_rows": report.missing_or_empty_rows,
            "conflicting_rows": report.conflicting_rows,
            "duplicate_rows": report.duplicate_rows,
            "eligible_rows": report.eligible_rows,
            "output_rows": report.output_rows,
        },
        "metadata": metadata,
    }


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy / pandas scalars."""

    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _format_summary(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    return (
        f"version: {summary['model_version']} ({summary['classifier']})\n"
        f"n_train={summary['n_train']} n_test={summary['n_test']}\n"
        f"accuracy={metrics['accuracy']:.4f}\n"
        f"macro_f1={metrics['macro_f1']:.4f}\n"
        f"weighted_f1={metrics['weighted_f1']:.4f}\n"
        f"artifact: {summary['paths']['joblib']}\n"
        f"metadata: {summary['paths']['metadata']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the baseline text classifier")
    parser.add_argument("--classifier", choices=VALID_CLASSIFIERS, default=None)
    parser.add_argument("--raw-csv", default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args(argv)

    summary = run_training(
        raw_csv_path=args.raw_csv,
        out_dir=args.out_dir,
        figures_dir=args.figures_dir,
        config_path=args.config,
        classifier=args.classifier,
    )
    print(_format_summary(summary), file=sys.stdout)
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Train, evaluate, and serialize the baseline text classifier."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from triage_ml.data.prepare import prepare_dataset, split_dataset
from triage_ml.models.artifact import (
    ArtifactPaths,
    build_metadata,
    write_classes,
    write_metadata,
)
from triage_ml.models.pipeline import (
    DEFAULT_LINEAR_SVC,
    DEFAULT_LOGREG,
    DEFAULT_TFIDF,
    VALID_CLASSIFIERS,
    build_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_CONFIG = Path("configs/training.yaml")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "training.yaml"
DEFAULT_RAW_CSV = Path("data/medical_tc_train.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports/figures")


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping")
    return config


def _coerce_tfidf(tfidf: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tfidf:
        return tfidf
    if not isinstance(tfidf, dict):
        raise ValueError("tfidf configuration must be a mapping")
    coerced = dict(tfidf)
    if isinstance(coerced.get("ngram_range"), list):
        coerced["ngram_range"] = tuple(coerced["ngram_range"])
    return coerced


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_fraction(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError(f"{name} must be a finite number between zero and one")
    return float(value)


def _classifier_config(config: dict[str, Any], name: str, random_state: int) -> dict[str, Any]:
    raw = config.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{name} configuration must be a mapping")
    return {**raw, "random_state": random_state}


def dataframe_fingerprint(data: pd.DataFrame) -> str:
    """Fingerprint ordered rows without persisting their clinical text."""

    digest = hashlib.sha256()
    for row in data[["text", "target"]].itertuples(index=False):
        text_digest = hashlib.sha256(str(row.text).encode("utf-8")).hexdigest()
        digest.update(f"{int(row.target)}:{text_digest}\n".encode())
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    if not (REPO_ROOT / "pyproject.toml").is_file():
        return "unknown", True
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    commit = commit_result.stdout.strip().lower()
    return (commit if len(commit) == 40 else "unknown", bool(status_result.stdout.strip()))


def _model_version(prepared_fingerprint: str, config_fingerprint: str) -> str:
    input_hash = hashlib.sha256(
        f"{prepared_fingerprint}:{config_fingerprint}".encode()
    ).hexdigest()[:12]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{input_hash}"


def compare_classifiers(
    train_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    random_state: int,
    selected_classifier: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compare candidates only on training folds and fix the final classifier."""

    folds = _require_int(config.get("cv_folds", 5), name="cv_folds", minimum=2)
    class_counts = train_df["target"].value_counts()
    if class_counts.empty or int(class_counts.min()) < folds:
        raise ValueError("each class must contain at least cv_folds training rows")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores: dict[str, dict[str, Any]] = {}
    for name in VALID_CLASSIFIERS:
        params = _classifier_config(config, name, random_state)
        pipeline = build_pipeline(
            name,
            tfidf=_coerce_tfidf(config.get("tfidf")),
            classifier_kwargs=params,
        )
        fold_scores = cross_val_score(
            pipeline,
            train_df["text"],
            train_df["target"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
        )
        if not np.isfinite(fold_scores).all():
            raise ValueError(f"cross-validation returned non-finite scores for {name}")
        scores[name] = {
            "fold_macro_f1": [float(score) for score in fold_scores],
            "mean_macro_f1": float(fold_scores.mean()),
            "std_macro_f1": float(fold_scores.std()),
        }

    best_classifier = max(VALID_CLASSIFIERS, key=lambda name: scores[name]["mean_macro_f1"])
    chosen = selected_classifier or best_classifier
    if chosen not in VALID_CLASSIFIERS:
        raise ValueError(f"Unsupported classifier {chosen!r}; valid: {VALID_CLASSIFIERS}")
    return chosen, {
        "metric": "macro_f1",
        "cv": "StratifiedKFold",
        "folds": folds,
        "candidates": scores,
        "best_classifier": best_classifier,
        "selected_classifier": chosen,
        "selection_policy": "explicit_override" if selected_classifier else "highest_mean_macro_f1",
        "test_set_used_for_selection": False,
    }


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "per_class": {
            str(label): {
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
    clf = pipeline.named_steps["clf"]
    tfidf: Any = pipeline.named_steps["tfidf"]
    if not hasattr(clf, "coef_"):
        return None
    feature_names = tfidf.get_feature_names_out()
    classes = list(getattr(clf, "classes_", labels))
    coefficients = clf.coef_
    if len(classes) == 2 and len(coefficients) == 1:
        class_coefficients = ((classes[0], -coefficients[0]), (classes[1], coefficients[0]))
    else:
        class_coefficients = zip(classes, coefficients, strict=True)
    fig, axes = plt.subplots(
        nrows=len(classes), ncols=1, figsize=(8, 1.6 * len(classes)), sharex=True
    )
    if len(classes) == 1:
        axes = [axes]
    for ax, (cls, coefs) in zip(axes, class_coefficients, strict=True):
        top = np.argsort(coefs)[-top_n:][::-1]
        tokens = [feature_names[index] for index in top]
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
    config_path: str | Path | None = None,
    classifier: str | None = None,
    sample_size: int | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run selection, final evaluation, and immutable artifact persistence."""

    selected_config = Path(config_path) if config_path is not None else PROJECT_CONFIG
    if config_path is None and not selected_config.is_file():
        selected_config = DEFAULT_CONFIG
    config = load_config(selected_config)
    sample = _require_int(
        config["sample_size"] if sample_size is None else sample_size,
        name="sample_size",
        minimum=1,
    )
    test_frac = _require_fraction(
        config["test_size"] if test_size is None else test_size, name="test_size"
    )
    seed = _require_int(
        config["random_state"] if random_state is None else random_state,
        name="random_state",
    )
    if classifier is not None and classifier not in VALID_CLASSIFIERS:
        raise ValueError(f"Unsupported classifier {classifier!r}; valid: {VALID_CLASSIFIERS}")

    raw_csv_path = Path(raw_csv_path)
    raw_csv_bytes = raw_csv_path.read_bytes()
    raw = pd.read_csv(io.BytesIO(raw_csv_bytes))
    canonical, report = prepare_dataset(raw, sample_size=sample, random_state=seed)
    train_df, test_df = split_dataset(canonical, test_size=test_frac, random_state=seed)
    labels = sorted(int(value) for value in canonical["target"].unique())
    label_mapping = {str(key): str(value) for key, value in config["label_mapping"].items()}
    if set(label_mapping) != {str(label) for label in labels}:
        raise ValueError("label_mapping must cover exactly the prepared dataset classes")

    classifier_name, selection = compare_classifiers(
        train_df,
        config,
        random_state=seed,
        selected_classifier=classifier,
    )
    classifier_params = _classifier_config(config, classifier_name, seed)
    pipeline = build_pipeline(
        classifier_name,
        tfidf=_coerce_tfidf(config.get("tfidf")),
        classifier_kwargs=classifier_params,
    )
    pipeline.fit(train_df["text"], train_df["target"])
    y_pred = pipeline.predict(test_df["text"])
    metrics = compute_metrics(test_df["target"], y_pred, labels)

    effective_tfidf = {**DEFAULT_TFIDF, **(_coerce_tfidf(config.get("tfidf")) or {})}
    effective_candidates = {
        name: {
            **(DEFAULT_LOGREG if name == "logreg" else DEFAULT_LINEAR_SVC),
            **_classifier_config(config, name, seed),
        }
        for name in VALID_CLASSIFIERS
    }
    configured_name = str(config.get("model_name", "triage_ml_tfidf_logreg"))
    model_name = (
        configured_name if classifier_name == "logreg" else f"triage_ml_tfidf_{classifier_name}"
    )
    effective_config = {
        "sample_size": sample,
        "test_size": test_frac,
        "random_state": seed,
        "cv_folds": selection["folds"],
        "language": str(config["language"]),
        "task_type": str(config["task_type"]),
        "label_mapping": label_mapping,
        "tfidf": effective_tfidf,
        "candidates": effective_candidates,
        "selected_classifier": classifier_name,
        "selection_policy": selection["selection_policy"],
        "model_name": model_name,
    }
    config_fingerprint = _sha256_json(effective_config)
    prepared_fingerprint = dataframe_fingerprint(canonical)
    fingerprints = {
        "raw_csv_sha256": hashlib.sha256(raw_csv_bytes).hexdigest(),
        "prepared_dataset_sha256": prepared_fingerprint,
        "train_split_sha256": dataframe_fingerprint(train_df),
        "test_split_sha256": dataframe_fingerprint(test_df),
        "config_sha256": config_fingerprint,
    }
    model_version = _model_version(prepared_fingerprint, config_fingerprint)
    paths = ArtifactPaths.for_version(out_dir, model_version)
    if paths.version_dir.exists():
        raise FileExistsError(f"artifact version already exists: {paths.version_dir}")
    figures_root = Path(figures_dir)
    version_figures_dir = figures_root / model_version
    if version_figures_dir.exists():
        raise FileExistsError(f"figure version already exists: {version_figures_dir}")
    staging_paths = ArtifactPaths.for_version(out_dir, f".{model_version}.tmp-{uuid.uuid4().hex}")
    staging_paths.ensure()
    git_commit, git_dirty = _git_state()
    classifier_defaults = DEFAULT_LOGREG if classifier_name == "logreg" else DEFAULT_LINEAR_SVC
    effective_classifier_params = {**classifier_defaults, **classifier_params}
    suffix = "lr" if classifier_name == "logreg" else classifier_name
    cm_path = version_figures_dir / f"08_confusion_matrix_{suffix}.png"
    top_path = version_figures_dir / f"08_top_features_{suffix}.png"
    figures_reserved = False
    try:
        version_figures_dir.mkdir(parents=True, exist_ok=False)
        figures_reserved = True
        joblib.dump(pipeline, staging_paths.joblib)
        persisted_classes = write_classes(staging_paths.classes, pipeline.classes_)
        metadata = build_metadata(
            model_version=model_version,
            model_name=model_name,
            task_type=str(config["task_type"]),
            language=str(config["language"]),
            classes=persisted_classes,
            label_mapping=label_mapping,
            random_state=seed,
            n_train=len(train_df),
            n_test=len(test_df),
            metrics=metrics,
            preprocessing={
                "vectorizer": "tfidf",
                "tfidf": effective_tfidf,
                "classifier": classifier_name,
                "classifier_params": effective_classifier_params,
            },
            selection=selection,
            dependency_versions={
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            git_commit=git_commit,
            git_dirty=git_dirty,
            fingerprints=fingerprints,
            joblib_path=staging_paths.joblib,
        )
        write_metadata(staging_paths.metadata, metadata)
        plot_confusion_matrix(test_df["target"], y_pred, labels, cm_path)
        top_path = plot_top_features(pipeline, labels, top_path)
        summary = {
            "model_version": model_version,
            "model_name": model_name,
            "classifier": classifier_name,
            "paths": {
                "joblib": str(paths.joblib),
                "classes": str(paths.classes),
                "metadata": str(paths.metadata),
                "summary": str(paths.version_dir / "summary.json"),
                "confusion_matrix": str(cm_path),
                "top_features": str(top_path) if top_path else None,
            },
            "selection": selection,
            "metrics": metrics,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "preparation_report": {
                "input_rows": report.input_rows,
                "missing_or_empty_rows": report.missing_or_empty_rows,
                "conflicting_texts": report.conflicting_texts,
                "conflicting_rows": report.conflicting_rows,
                "duplicate_rows": report.duplicate_rows,
                "eligible_rows": report.eligible_rows,
                "output_rows": report.output_rows,
            },
            "metadata": metadata,
        }
        (staging_paths.version_dir / "summary.json").write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        staging_paths.version_dir.rename(paths.version_dir)
    except Exception:
        shutil.rmtree(staging_paths.version_dir, ignore_errors=True)
        if figures_reserved:
            shutil.rmtree(version_figures_dir, ignore_errors=True)
        raise
    return summary


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _format_summary(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    return (
        f"version: {summary['model_version']} ({summary['classifier']})\n"
        f"n_train={summary['n_train']} n_test={summary['n_test']}\n"
        f"accuracy={metrics['accuracy']:.4f}\n"
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}\n"
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
    parser.add_argument("--config", default=None)
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
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

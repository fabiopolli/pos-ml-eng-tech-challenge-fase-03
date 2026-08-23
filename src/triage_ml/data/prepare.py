"""Reproducible preparation of the Medical Abstracts dataset."""

import math
from dataclasses import dataclass
from numbers import Integral

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_COLUMNS = {"medical_abstract": "text", "condition_label": "target"}
CANONICAL_COLUMNS = ["text", "target"]
VALID_TARGETS = frozenset(range(1, 6))


@dataclass(frozen=True)
class PreparationReport:
    """Counts recorded while producing the canonical dataset."""

    input_rows: int
    missing_or_empty_rows: int
    conflicting_texts: int
    conflicting_rows: int
    duplicate_rows: int
    eligible_rows: int
    output_rows: int


def _canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {names}")

    data = raw.rename(columns=RAW_COLUMNS)[CANONICAL_COLUMNS].copy()
    non_null_text = data["text"].notna()
    if not data.loc[non_null_text, "text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("medical_abstract must contain only strings or null values")
    data["text"] = data["text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()

    raw_target = data["target"]
    numeric_target = pd.to_numeric(raw_target, errors="coerce")
    invalid_target = raw_target.notna() & (
        raw_target.map(lambda value: isinstance(value, bool))
        | numeric_target.isna()
        | ~numeric_target.map(
            lambda value: math.isfinite(float(value)) if pd.notna(value) else True
        )
        | numeric_target.map(
            lambda value: float(value).is_integer() if pd.notna(value) else True
        ).eq(False)
    )
    if invalid_target.any():
        raise ValueError("condition_label must contain finite integer values or null values")
    data["target"] = numeric_target
    return data


def _text_key(text: pd.Series) -> pd.Series:
    """Match the model's case-insensitive view when grouping equivalent texts."""

    return text.str.normalize("NFKC").str.casefold()


def _stratified_sample(data: pd.DataFrame, *, size: int, random_state: int) -> pd.DataFrame:
    """Select exact deterministic per-class quotas without requiring a large complement."""

    counts = data["target"].value_counts().sort_index()
    ideal = counts * (size / len(data))
    quotas = ideal.map(math.floor).astype(int)
    remaining = size - int(quotas.sum())
    order = sorted(counts.index, key=lambda label: (-(ideal[label] - quotas[label]), label))
    for label in order:
        if remaining == 0:
            break
        if quotas[label] < counts[label]:
            quotas[label] += 1
            remaining -= 1
    if remaining != 0 or (quotas == 0).any():
        raise ValueError("class distribution cannot support the requested stratified sample")
    return pd.concat(
        [
            group.sample(n=int(quotas[label]), random_state=random_state)
            for label, group in data.groupby("target", sort=True)
        ],
        ignore_index=True,
    )


def prepare_dataset(
    raw: pd.DataFrame,
    *,
    sample_size: int = 5_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PreparationReport]:
    """Create a unique, conflict-free, stratified canonical sample.

    Exact normalized texts associated with multiple targets are excluded because
    automatically choosing one of their labels would not be defensible.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size must be an integer")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")
    if not 2_000 <= sample_size <= 5_000:
        raise ValueError("sample_size must respect the project contract: 2,000 to 5,000")

    data = _canonicalize(raw)
    valid = data["text"].notna() & data["text"].ne("") & data["target"].notna()
    missing_or_empty_rows = int((~valid).sum())
    data = data.loc[valid].copy()
    data["target"] = data["target"].astype(int)
    unknown_targets = set(data["target"]) - VALID_TARGETS
    if unknown_targets:
        raise ValueError(f"condition_label contains unsupported labels: {sorted(unknown_targets)}")
    data["_text_key"] = _text_key(data["text"])

    targets_per_text = data.groupby("_text_key", sort=False)["target"].nunique()
    conflicting_texts = targets_per_text[targets_per_text > 1].index
    conflicting_mask = data["_text_key"].isin(conflicting_texts)
    conflicting_rows = int(conflicting_mask.sum())
    data = data.loc[~conflicting_mask].copy()

    before_deduplication = len(data)
    data = data.drop_duplicates(subset="_text_key", keep="first")
    duplicate_rows = before_deduplication - len(data)
    eligible_rows = len(data)

    if eligible_rows < sample_size:
        raise ValueError(
            f"Only {eligible_rows} eligible rows remain; cannot create a {sample_size}-row sample"
        )

    if eligible_rows > sample_size:
        data = _stratified_sample(data, size=sample_size, random_state=random_state)

    data = (
        data.drop(columns="_text_key")
        .sort_values(["target", "text"], kind="stable")
        .reset_index(drop=True)
    )
    report = PreparationReport(
        input_rows=len(raw),
        missing_or_empty_rows=missing_or_empty_rows,
        conflicting_texts=len(conflicting_texts),
        conflicting_rows=conflicting_rows,
        duplicate_rows=duplicate_rows,
        eligible_rows=eligible_rows,
        output_rows=len(data),
    )
    return data, report


def split_dataset(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified split and assert that exact text cannot leak."""
    if list(data.columns) != CANONICAL_COLUMNS:
        raise ValueError(f"Expected canonical columns in order: {CANONICAL_COLUMNS}")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be a finite number between zero and one")
    if data.empty or data.isna().any().any():
        raise ValueError("Canonical dataset must be non-empty and cannot contain null values")
    if not data["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Canonical text values must be non-empty strings")
    if (
        not data["target"]
        .map(lambda value: isinstance(value, Integral) and not isinstance(value, bool))
        .all()
    ):
        raise ValueError("Canonical targets must be integers")
    if not set(data["target"]).issubset(VALID_TARGETS):
        raise ValueError("Canonical targets contain unsupported labels")
    text_keys = _text_key(data["text"].astype("string"))
    if text_keys.duplicated().any():
        raise ValueError("Dataset contains duplicate texts; prepare it before splitting")

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["target"],
    )
    overlap = set(_text_key(train["text"].astype("string"))) & set(
        _text_key(test["text"].astype("string"))
    )
    if overlap:
        raise AssertionError(f"Text leakage detected for {len(overlap)} texts")
    return train.reset_index(drop=True), test.reset_index(drop=True)

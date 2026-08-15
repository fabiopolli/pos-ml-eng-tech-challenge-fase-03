"""Reproducible preparation of the Medical Abstracts dataset."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_COLUMNS = {"medical_abstract": "text", "condition_label": "target"}
CANONICAL_COLUMNS = ["text", "target"]


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
    data["text"] = data["text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    return data


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
    if not 2_000 <= sample_size <= 5_000:
        raise ValueError("sample_size must respect the project contract: 2,000 to 5,000")

    data = _canonicalize(raw)
    valid = data["text"].notna() & data["text"].ne("") & data["target"].notna()
    missing_or_empty_rows = int((~valid).sum())
    data = data.loc[valid].copy()
    data["target"] = data["target"].astype(int)

    targets_per_text = data.groupby("text", sort=False)["target"].nunique()
    conflicting_texts = targets_per_text[targets_per_text > 1].index
    conflicting_mask = data["text"].isin(conflicting_texts)
    conflicting_rows = int(conflicting_mask.sum())
    data = data.loc[~conflicting_mask].copy()

    before_deduplication = len(data)
    data = data.drop_duplicates(subset="text", keep="first")
    duplicate_rows = before_deduplication - len(data)
    eligible_rows = len(data)

    if eligible_rows < sample_size:
        raise ValueError(
            f"Only {eligible_rows} eligible rows remain; cannot create a {sample_size}-row sample"
        )

    if eligible_rows > sample_size:
        data, _ = train_test_split(
            data,
            train_size=sample_size,
            random_state=random_state,
            stratify=data["target"],
        )

    data = data.sort_values(["target", "text"], kind="stable").reset_index(drop=True)
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
    if data["text"].duplicated().any():
        raise ValueError("Dataset contains duplicate texts; prepare it before splitting")

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["target"],
    )
    overlap = set(train["text"]) & set(test["text"])
    if overlap:
        raise AssertionError(f"Text leakage detected for {len(overlap)} texts")
    return train.reset_index(drop=True), test.reset_index(drop=True)

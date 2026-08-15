import pandas as pd
import pytest

from triage_ml.data.prepare import prepare_dataset, split_dataset


def _raw_dataset(rows_per_label: int = 600) -> pd.DataFrame:
    rows = [
        {"medical_abstract": f"Unique abstract {label} {index}", "condition_label": label}
        for label in range(1, 6)
        for index in range(rows_per_label)
    ]
    rows.extend(
        [
            {"medical_abstract": "  duplicate   same label ", "condition_label": 1},
            {"medical_abstract": "duplicate same label", "condition_label": 1},
            {"medical_abstract": "conflicting text", "condition_label": 1},
            {"medical_abstract": "conflicting text", "condition_label": 2},
            {"medical_abstract": None, "condition_label": 3},
        ]
    )
    return pd.DataFrame(rows)


def test_prepare_dataset_enforces_canonical_contract() -> None:
    prepared, report = prepare_dataset(_raw_dataset(), sample_size=2_000)

    assert list(prepared.columns) == ["text", "target"]
    assert len(prepared) == 2_000
    assert not prepared["text"].duplicated().any()
    assert prepared.groupby("text")["target"].nunique().max() == 1
    assert report.conflicting_texts == 1
    assert report.conflicting_rows == 2
    assert report.missing_or_empty_rows == 1
    assert report.duplicate_rows == 1


def test_prepare_dataset_is_reproducible() -> None:
    first, _ = prepare_dataset(_raw_dataset(), sample_size=2_000, random_state=42)
    second, _ = prepare_dataset(_raw_dataset(), sample_size=2_000, random_state=42)

    pd.testing.assert_frame_equal(first, second)


def test_split_is_stratified_and_has_no_text_leakage() -> None:
    prepared, _ = prepare_dataset(_raw_dataset(), sample_size=2_000)
    train, test = split_dataset(prepared)

    assert len(train) == 1_600
    assert len(test) == 400
    assert set(train["text"]).isdisjoint(test["text"])
    assert train["target"].value_counts().to_dict() == {1: 320, 2: 320, 3: 320, 4: 320, 5: 320}
    assert test["target"].value_counts().to_dict() == {1: 80, 2: 80, 3: 80, 4: 80, 5: 80}


def test_split_rejects_duplicate_texts() -> None:
    invalid = pd.DataFrame({"text": ["same", "same"], "target": [1, 1]})

    with pytest.raises(ValueError, match="duplicate texts"):
        split_dataset(invalid)

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


@pytest.mark.parametrize("label", [True, 1.5, float("inf"), "not-a-label", 6])
def test_prepare_dataset_rejects_invalid_labels(label: object) -> None:
    raw = _raw_dataset()
    raw["condition_label"] = raw["condition_label"].astype(object)
    raw.loc[0, "condition_label"] = label

    with pytest.raises(ValueError, match="condition_label"):
        prepare_dataset(raw, sample_size=2_000)


def test_prepare_dataset_rejects_non_string_text() -> None:
    raw = _raw_dataset()
    raw["medical_abstract"] = raw["medical_abstract"].astype(object)
    raw.loc[0, "medical_abstract"] = 123

    with pytest.raises(ValueError, match="medical_abstract"):
        prepare_dataset(raw, sample_size=2_000)


def test_prepare_dataset_groups_unicode_and_case_equivalent_texts() -> None:
    raw = _raw_dataset()
    raw.loc[0, "medical_abstract"] = "Kidney Disease"
    raw.loc[1, "medical_abstract"] = "ＫIDNEY disease"
    raw.loc[1, "condition_label"] = 2

    _, report = prepare_dataset(raw, sample_size=2_000)

    assert report.conflicting_texts == 2
    assert report.conflicting_rows == 4


@pytest.mark.parametrize("test_size", [True, 0, 1, float("nan"), "0.2"])
def test_split_rejects_invalid_test_size(test_size: object) -> None:
    prepared, _ = prepare_dataset(_raw_dataset(), sample_size=2_000)

    with pytest.raises(ValueError, match="test_size"):
        split_dataset(prepared, test_size=test_size)  # type: ignore[arg-type]

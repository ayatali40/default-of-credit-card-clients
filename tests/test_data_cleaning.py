"""Tests for cleaning rules and data quality reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    clean_categorical_codes,
    clean_dataset,
    clean_feature_values,
    count_unexpected_codes,
    data_quality_report,
)


def test_unexpected_codes_are_mapped_without_dropping_rows(
    synthetic_raw_df: pd.DataFrame,
):
    before = len(synthetic_raw_df)
    unexpected = count_unexpected_codes(synthetic_raw_df)
    assert unexpected["EDUCATION"]  # fixture intentionally contains 0, 5, 6
    assert unexpected["MARRIAGE"]

    cleaned = clean_dataset(synthetic_raw_df.copy())

    assert len(cleaned) == before
    assert set(cleaned["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(cleaned["MARRIAGE"].unique()) <= {1, 2, 3}


def test_cleaning_does_not_mutate_input(synthetic_raw_df: pd.DataFrame):
    original = synthetic_raw_df.copy(deep=True)
    clean_dataset(synthetic_raw_df)
    pd.testing.assert_frame_equal(synthetic_raw_df, original)


def test_duplicate_rows_are_removed(synthetic_raw_df: pd.DataFrame):
    duplicated = pd.concat(
        [synthetic_raw_df, synthetic_raw_df.iloc[[0]]], ignore_index=True
    )

    cleaned = clean_dataset(duplicated)

    assert len(cleaned) == len(synthetic_raw_df)


def test_duplicate_ids_are_preserved(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.copy()
    frame.loc[1, "ID"] = frame.loc[0, "ID"]

    cleaned = clean_dataset(frame)

    assert cleaned["ID"].duplicated().sum() == 1
    assert len(cleaned) == len(frame)


def test_invalid_target_raises(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.copy()
    frame.loc[0, "default"] = 2

    with pytest.raises(ValueError, match="outside"):
        clean_dataset(frame)


def test_unexpected_sex_code_raises(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.copy()
    frame.loc[0, "SEX"] = 9

    with pytest.raises(ValueError, match="SEX"):
        clean_categorical_codes(frame)


def test_clean_feature_values_does_not_require_target(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.drop(columns=["default"])

    cleaned = clean_feature_values(frame)

    assert "default" not in cleaned.columns
    assert set(cleaned["EDUCATION"].unique()) <= {1, 2, 3, 4}


def test_infinite_numeric_value_is_rejected(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.copy()
    frame["BILL_AMT1"] = frame["BILL_AMT1"].astype(float)
    frame.loc[0, "BILL_AMT1"] = np.inf

    with pytest.raises(ValueError, match="infinite"):
        clean_feature_values(frame)


def test_missing_categorical_values_are_not_treated_as_invalid_codes(
    synthetic_raw_df: pd.DataFrame,
):
    frame = synthetic_raw_df.copy()
    frame["EDUCATION"] = frame["EDUCATION"].astype(float)
    frame["MARRIAGE"] = frame["MARRIAGE"].astype(float)
    frame["SEX"] = frame["SEX"].astype(float)
    frame.loc[0, ["EDUCATION", "MARRIAGE", "SEX"]] = np.nan

    cleaned = clean_categorical_codes(frame)

    assert cleaned.loc[0, ["EDUCATION", "MARRIAGE", "SEX"]].isna().all()


def test_clean_feature_values_keeps_nullable_missing_values(
    synthetic_raw_df: pd.DataFrame,
):
    frame = synthetic_raw_df.drop(columns=["default"]).copy()
    frame["BILL_AMT1"] = frame["BILL_AMT1"].astype("Int64")
    frame.loc[0, "BILL_AMT1"] = pd.NA
    original = frame.copy(deep=True)

    cleaned = clean_feature_values(frame)

    assert pd.isna(cleaned.loc[0, "BILL_AMT1"])
    assert cleaned["BILL_AMT1"].isna().sum() == 1
    pd.testing.assert_frame_equal(frame, original)


def test_negative_infinity_is_rejected(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.copy()
    frame["PAY_AMT1"] = frame["PAY_AMT1"].astype(float)
    frame.loc[0, "PAY_AMT1"] = -np.inf

    with pytest.raises(ValueError, match="infinite"):
        clean_feature_values(frame)


def test_missing_target_is_rejected(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.copy()
    frame.loc[0, "default"] = np.nan

    with pytest.raises(ValueError, match="missing, non-numeric"):
        clean_dataset(frame)


def test_non_numeric_target_is_rejected(synthetic_raw_df: pd.DataFrame):
    frame = synthetic_raw_df.copy()
    frame["default"] = frame["default"].astype(object)
    frame.loc[0, "default"] = "yes"

    with pytest.raises(ValueError, match="missing, non-numeric"):
        clean_dataset(frame)


def test_unexpected_code_counts_do_not_truncate_non_integer_values(
    synthetic_raw_df: pd.DataFrame,
):
    frame = synthetic_raw_df.copy()
    frame["EDUCATION"] = frame["EDUCATION"].astype(float)
    frame.loc[0, "EDUCATION"] = 1.5

    report = count_unexpected_codes(frame)

    assert report["EDUCATION"][1.5] == 1
    assert 1 not in report["EDUCATION"]


def test_data_quality_report_contains_expected_sections(synthetic_raw_df: pd.DataFrame):
    report = data_quality_report(synthetic_raw_df)

    assert report["n_rows"] == len(synthetic_raw_df)
    assert report["n_columns"] == synthetic_raw_df.shape[1]
    assert report["missing_total"] == 0
    assert report["duplicate_ids"] == 0
    assert report["target_distribution"] == {"0": 252, "1": 108}
    assert 0 < report["default_rate"] < 1
    assert "EDUCATION" in report["unexpected_category_codes"]

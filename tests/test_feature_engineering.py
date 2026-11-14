"""Tests for engineered features, zero-limit handling, and target isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ENGINEERED_FEATURES,
    engineer_features,
    safe_divide,
)


def test_all_engineered_features_are_created(synthetic_cleaned_df: pd.DataFrame):
    engineered = engineer_features(synthetic_cleaned_df)

    for column in ENGINEERED_FEATURES:
        assert column in engineered.columns


def test_feature_engineering_does_not_mutate_input(synthetic_cleaned_df: pd.DataFrame):
    original = synthetic_cleaned_df.copy(deep=True)

    engineer_features(synthetic_cleaned_df)

    pd.testing.assert_frame_equal(synthetic_cleaned_df, original)


def test_average_and_total_values_are_correct(synthetic_cleaned_df: pd.DataFrame):
    row = synthetic_cleaned_df.iloc[[0]].copy()
    for index in range(1, 7):
        row[f"BILL_AMT{index}"] = index
        row[f"PAY_AMT{index}"] = index * 10
        row[f"PAY_{index}"] = index - 1

    engineered = engineer_features(row).iloc[0]

    assert engineered["BILL_AVG"] == pytest.approx(3.5)
    assert engineered["BILL_TOTAL"] == pytest.approx(21.0)
    assert engineered["PAY_AVG"] == pytest.approx(35.0)
    assert engineered["PAY_TOTAL"] == pytest.approx(210.0)
    assert engineered["NUM_DELAYED_MONTHS"] == 5  # statuses 1..5 are > 0
    assert engineered["NUM_SEVERE_DELAY_MONTHS"] == 4  # statuses 2..5 are > 1
    assert engineered["MAX_DELAY_STATUS"] == 5
    assert engineered["ZERO_PAYMENT_MONTHS"] == 0


def test_zero_credit_limit_produces_finite_utilization(
    synthetic_cleaned_df: pd.DataFrame,
):
    frame = synthetic_cleaned_df.copy()
    frame.loc[:, "LIMIT_BAL"] = 0

    engineered = engineer_features(frame)

    assert (engineered["UTILIZATION_AVG"] == 0).all()
    assert (engineered["UTILIZATION_LAST"] == 0).all()
    assert np.isfinite(engineered["UTILIZATION_AVG"]).all()


def test_zero_bills_do_not_create_infinite_ratios(synthetic_cleaned_df: pd.DataFrame):
    frame = synthetic_cleaned_df.copy()
    for index in range(1, 7):
        frame[f"BILL_AMT{index}"] = 0
        frame[f"PAY_AMT{index}"] = 0
    frame["LIMIT_BAL"] = 0

    engineered = engineer_features(frame)
    numeric_values = engineered.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    assert np.isfinite(numeric_values).all()
    assert (engineered["PAYMENT_TO_BILL_RATIO"] == 0).all()


def test_target_is_not_used_in_feature_engineering(synthetic_cleaned_df: pd.DataFrame):
    altered = synthetic_cleaned_df.copy()
    altered["default"] = 1 - altered["default"]

    original_features = engineer_features(synthetic_cleaned_df)[ENGINEERED_FEATURES]
    altered_features = engineer_features(altered)[ENGINEERED_FEATURES]

    pd.testing.assert_frame_equal(original_features, altered_features)


def test_safe_divide_handles_zero_and_negative_denominators():
    numerator = pd.Series([10.0, 5.0, -5.0])
    denominator = pd.Series([0.0, 2.0, -2.0])

    result = safe_divide(numerator, denominator)

    assert result.tolist() == [0.0, 2.5, 2.5]
    assert np.isfinite(result).all()


def test_partial_six_month_history_stays_missing(synthetic_cleaned_df: pd.DataFrame):
    frame = synthetic_cleaned_df.copy()
    for column in ["BILL_AMT3", "PAY_2", "PAY_AMT4"]:
        frame[column] = frame[column].astype(float)
        frame.loc[0, column] = np.nan

    engineered = engineer_features(frame)
    row = engineered.iloc[0]

    missing_features = [
        "BILL_AVG",
        "BILL_TOTAL",
        "PAY_AVG",
        "PAY_TOTAL",
        "NUM_DELAYED_MONTHS",
        "NUM_SEVERE_DELAY_MONTHS",
        "MAX_DELAY_STATUS",
        "AVG_DELAY_STATUS",
        "UTILIZATION_AVG",
        "PAYMENT_TO_BILL_RATIO",
        "ZERO_PAYMENT_MONTHS",
    ]
    for column in missing_features:
        assert pd.isna(row[column]), column


def test_safe_divide_propagates_missing_inputs():
    result = safe_divide(
        pd.Series([np.nan, 6.0, 3.0]),
        pd.Series([2.0, np.nan, 0.0]),
    )

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 0.0


def test_extreme_financial_values_produce_finite_features(
    synthetic_cleaned_df: pd.DataFrame,
):
    frame = synthetic_cleaned_df.copy()
    frame["LIMIT_BAL"] = 1_000_000
    frame["BILL_AMT1"] = 1_000_000_000_000
    frame["PAY_AMT1"] = 1_000_000_000_000

    engineered = engineer_features(frame)

    assert np.isfinite(engineered["UTILIZATION_LAST"]).all()
    assert (engineered["UTILIZATION_LAST"] > 0).all()


def test_missing_required_column_raises(synthetic_cleaned_df: pd.DataFrame):
    frame = synthetic_cleaned_df.drop(columns=["BILL_AMT3"])

    with pytest.raises(ValueError, match="missing columns"):
        engineer_features(frame)

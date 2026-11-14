"""Tests for stratified splitting and leakage-safe preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    build_model_pipeline,
    build_preprocessor,
    class_distribution,
    get_feature_lists,
    split_dataset,
)


def test_split_sizes_and_stratification(synthetic_cleaned_df: pd.DataFrame):
    train, validation, test = split_dataset(synthetic_cleaned_df)

    total = len(synthetic_cleaned_df)
    assert len(train) == pytest.approx(0.70 * total, abs=1)
    assert len(validation) == pytest.approx(0.15 * total, abs=1)
    assert len(test) == pytest.approx(0.15 * total, abs=1)
    assert len(train) + len(validation) + len(test) == total

    overall_rate = synthetic_cleaned_df["default"].mean()
    for split in (train, validation, test):
        assert split["default"].mean() == pytest.approx(overall_rate, abs=0.03)


def test_splits_do_not_share_customers(synthetic_cleaned_df: pd.DataFrame):
    train, validation, test = split_dataset(synthetic_cleaned_df)

    assert set(train["ID"]).isdisjoint(validation["ID"])
    assert set(train["ID"]).isdisjoint(test["ID"])
    assert set(validation["ID"]).isdisjoint(test["ID"])


def test_split_handles_duplicated_input_index_labels(
    synthetic_cleaned_df: pd.DataFrame,
):
    frame = synthetic_cleaned_df.copy()
    midpoint = len(frame) // 2
    frame.index = [0] * midpoint + [1] * (len(frame) - midpoint)

    train, validation, test = split_dataset(frame)

    assert len(train) + len(validation) + len(test) == len(frame)
    assert set(train["ID"]).isdisjoint(validation["ID"])
    assert set(train["ID"]).isdisjoint(test["ID"])
    assert set(validation["ID"]).isdisjoint(test["ID"])


def test_unexpected_non_numeric_column_is_rejected(
    synthetic_engineered_df: pd.DataFrame,
):
    frame = synthetic_engineered_df.copy()
    frame["FREE_TEXT"] = "note"

    with pytest.raises(ValueError, match="neither numeric"):
        get_feature_lists(frame)


def test_missing_categorical_feature_is_rejected(
    synthetic_engineered_df: pd.DataFrame,
):
    frame = synthetic_engineered_df.drop(columns=["SEX"])

    with pytest.raises(ValueError, match="Expected categorical features are missing"):
        get_feature_lists(frame)


def test_empty_class_distribution_is_rejected():
    with pytest.raises(ValueError, match="empty series"):
        class_distribution(pd.Series([], dtype=int))


def test_invalid_split_fractions_raise(synthetic_cleaned_df: pd.DataFrame):
    with pytest.raises(ValueError):
        split_dataset(synthetic_cleaned_df, test_size=0.6, validation_size=0.5)
    with pytest.raises(ValueError):
        split_dataset(synthetic_cleaned_df, test_size=0)


def test_feature_lists_exclude_id_and_target(synthetic_engineered_df: pd.DataFrame):
    numeric, categorical = get_feature_lists(synthetic_engineered_df)

    assert "ID" not in numeric and "ID" not in categorical
    assert "default" not in numeric and "default" not in categorical
    assert categorical == ["SEX", "EDUCATION", "MARRIAGE"]
    assert "LIMIT_BAL" in numeric
    assert "UTILIZATION_AVG" in numeric


def test_imputation_and_scaling_are_fitted_on_training_only():
    X_train = pd.DataFrame(
        {
            "AGE": [1.0, 2.0, 3.0, np.nan],
            "SEX": [1, 2, 1, 2],
        }
    )
    X_validation = pd.DataFrame({"AGE": [1000.0, 2000.0], "SEX": [1, 2]})

    preprocessor = build_preprocessor(["AGE"], ["SEX"])
    preprocessor.fit(X_train)

    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]

    # A combined train+validation median would be 3.0; train-only median is 2.0.
    assert imputer.statistics_[0] == pytest.approx(2.0)
    assert scaler.mean_[0] == pytest.approx(2.0)

    transformed_validation = preprocessor.transform(X_validation)
    assert np.isfinite(transformed_validation).all()


def test_validation_data_cannot_change_imputation_statistics():
    X_train = pd.DataFrame({"AGE": [30.0, np.nan, 40.0], "SEX": [1, 1, 2]})
    X_validation = pd.DataFrame({"AGE": [10_000.0], "SEX": [1]})

    preprocessor = build_preprocessor(["AGE"], ["SEX"])
    preprocessor.fit(X_train)
    train_only_statistic = (
        preprocessor.named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_[0]
    )

    contaminated = build_preprocessor(["AGE"], ["SEX"])
    contaminated.fit(pd.concat([X_train, X_validation], ignore_index=True))
    contaminated_statistic = (
        contaminated.named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_[0]
    )

    assert train_only_statistic == pytest.approx(35.0)
    assert train_only_statistic != contaminated_statistic


def test_unseen_categories_are_ignored():
    X_train = pd.DataFrame({"AGE": [20.0, 30.0, 40.0], "SEX": [1, 2, 1]})
    X_test = pd.DataFrame({"AGE": [25.0], "SEX": [7]})

    preprocessor = build_preprocessor(["AGE"], ["SEX"])
    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_test)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_model_pipeline_fits_and_predicts(synthetic_engineered_df: pd.DataFrame):
    train, _, _ = split_dataset(synthetic_engineered_df)
    numeric, categorical = get_feature_lists(train)
    features = [column for column in train.columns if column not in {"ID", "default"}]

    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42), numeric, categorical
    )
    pipeline.fit(train[features], train["default"])
    probabilities = pipeline.predict_proba(train[features])[:, 1]

    assert len(probabilities) == len(train)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_class_distribution(synthetic_cleaned_df: pd.DataFrame):
    distribution = class_distribution(synthetic_cleaned_df["default"])

    assert list(distribution["default"]) == [0, 1]
    assert distribution["count"].sum() == len(synthetic_cleaned_df)
    assert distribution["proportion"].sum() == pytest.approx(1.0)

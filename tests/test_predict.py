"""Tests for artefact validation and the prediction path."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.predict import (
    PredictionError,
    _validate_artifact,
    load_model_artifact,
    predict_customers,
    validate_customer_records,
)
from src.preprocessing import (
    build_model_pipeline,
    get_feature_lists,
    split_dataset,
)


def _fitted_artifact(
    synthetic_engineered_df: pd.DataFrame, threshold: float = 0.5
) -> dict:
    train, _, _ = split_dataset(synthetic_engineered_df)
    numeric, categorical = get_feature_lists(train)
    features = [column for column in train.columns if column not in {"ID", "default"}]
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42), numeric, categorical
    )
    pipeline.fit(train[features], train["default"])
    return {"pipeline": pipeline, "threshold": threshold, "feature_columns": features}


def test_load_model_artifact_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        load_model_artifact(tmp_path / "missing.joblib")


def test_load_model_artifact_rejects_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"this is not a joblib artifact")

    with pytest.raises(PredictionError, match="Could not load"):
        load_model_artifact(path)


def test_load_model_artifact_requires_feature_columns(tmp_path):
    path = tmp_path / "artifact.joblib"
    joblib.dump({"pipeline": LogisticRegression(), "threshold": 0.5}, path)

    with pytest.raises(PredictionError, match="feature_columns"):
        load_model_artifact(path)


def test_validate_artifact_rejects_invalid_threshold(synthetic_engineered_df):
    artifact = _fitted_artifact(synthetic_engineered_df)
    artifact["threshold"] = 1.5

    with pytest.raises(PredictionError, match=r"\[0, 1\]"):
        _validate_artifact(artifact)


def test_validate_artifact_rejects_non_probability_pipeline():
    artifact = {
        "pipeline": LogisticRegression(),
        "threshold": 0.5,
        "feature_columns": ["LIMIT_BAL"],
    }
    artifact["pipeline"] = object()

    with pytest.raises(PredictionError, match="predict_proba"):
        _validate_artifact(artifact)


def test_validate_customer_records_ignores_extra_columns(synthetic_raw_df):
    records = synthetic_raw_df.head(5).copy()
    records["Notes"] = "operational metadata"
    records = records.rename(columns={"PAY_1": "PAY_0"})

    validated = validate_customer_records(records)

    assert "Notes" not in validated.columns
    assert "default" not in validated.columns
    assert "PAY_0" not in validated.columns
    assert "PAY_1" in validated.columns
    assert list(validated["ID"]) == list(records["ID"])


def test_validate_customer_records_generates_ids(synthetic_raw_df):
    records = synthetic_raw_df.drop(columns=["ID"]).head(3)

    validated = validate_customer_records(records)

    assert list(validated["ID"]) == ["Customer 1", "Customer 2", "Customer 3"]


def test_validate_customer_records_reports_missing_columns(synthetic_raw_df):
    records = synthetic_raw_df.drop(columns=["LIMIT_BAL"])

    with pytest.raises(PredictionError, match="LIMIT_BAL"):
        validate_customer_records(records)


def test_predict_customers_scores_every_record(
    synthetic_raw_df, synthetic_engineered_df
):
    artifact = _fitted_artifact(synthetic_engineered_df, threshold=0.5)
    records = synthetic_raw_df.head(20)

    output = predict_customers(records, artifact)

    assert len(output) == len(records)
    assert list(output.columns) == ["ID", "Default Probability", "Prediction"]
    assert output["Default Probability"].between(0, 1).all()
    assert set(output["Prediction"]) <= {"Default", "No Default"}


def test_predict_customers_finds_positive_class_by_label(
    synthetic_raw_df, synthetic_engineered_df
):
    class ReversedClassesEstimator:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.8), np.full(len(X), 0.2)])

    artifact = {
        "pipeline": ReversedClassesEstimator(),
        "threshold": 0.5,
        "feature_columns": ["LIMIT_BAL"],
    }

    output = predict_customers(synthetic_raw_df.head(4), artifact)

    assert np.allclose(output["Default Probability"], 0.8)
    assert (output["Prediction"] == "Default").all()


def test_predict_customers_rejects_non_numeric_records(
    synthetic_raw_df, synthetic_engineered_df
):
    artifact = _fitted_artifact(synthetic_engineered_df)
    records = synthetic_raw_df.head(2).copy()
    records["AGE"] = ["unknown", 30]

    with pytest.raises(PredictionError, match="Invalid customer record"):
        predict_customers(records, artifact)


def test_predict_customers_rejects_duplicate_standardized_columns(
    synthetic_raw_df, synthetic_engineered_df
):
    artifact = _fitted_artifact(synthetic_engineered_df)
    records = synthetic_raw_df.head(2).copy()
    records["PAY_0"] = 0

    with pytest.raises(PredictionError, match="Duplicate column names"):
        predict_customers(records, artifact)


def test_predict_customers_requires_documented_features(
    synthetic_raw_df, synthetic_engineered_df
):
    artifact = _fitted_artifact(synthetic_engineered_df)
    artifact["feature_columns"] = list(artifact["feature_columns"]) + ["NOT_A_FEATURE"]

    with pytest.raises(PredictionError, match="NOT_A_FEATURE"):
        predict_customers(synthetic_raw_df.head(2), artifact)


def test_predict_customers_does_not_modify_input(
    synthetic_raw_df, synthetic_engineered_df
):
    artifact = _fitted_artifact(synthetic_engineered_df)
    records = synthetic_raw_df.head(5).copy()
    original = records.copy(deep=True)

    predict_customers(records, artifact)

    pd.testing.assert_frame_equal(records, original)

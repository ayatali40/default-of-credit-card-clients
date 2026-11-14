"""Integration tests for training, persistence, and prediction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_loader import ID_COLUMN
from src.predict import (
    PredictionError,
    load_model_artifact,
    predict_customers,
)
from src.train import run_training_pipeline


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, synthetic_raw_df):
    base = tmp_path_factory.mktemp("training")
    dataset = base / "synthetic_credit.csv"
    synthetic_raw_df.to_csv(dataset, index=False)

    result = run_training_pipeline(
        dataset_path=dataset,
        reports_dir=base / "reports",
        images_dir=base / "images",
        models_dir=base / "models",
        model_names=["Logistic Regression", "Decision Tree"],
        cv_folds=2,
        tune=False,
        run_imbalance_experiments=False,
        random_state=42,
    )
    return result, base


def test_training_generates_required_artefacts(trained_run):
    _, base = trained_run

    assert (base / "models" / "final_model.joblib").exists()
    assert (base / "models" / "model_metadata.json").exists()
    assert (base / "reports" / "model_comparison.csv").exists()
    assert (base / "reports" / "classification_report.csv").exists()
    assert (base / "reports" / "final_test_metrics.json").exists()
    assert (base / "reports" / "threshold_analysis.csv").exists()
    assert (base / "reports" / "training_summary.json").exists()
    assert (base / "reports" / "data_quality_summary.json").exists()
    assert (base / "reports" / "selection_policy.json").exists()
    assert (base / "reports" / "subgroup_performance.csv").exists()
    assert (base / "reports" / "fairness_analysis.json").exists()
    assert (base / "reports" / "feature_importance.csv").exists()
    assert (base / "reports" / "feature_importance_model_based.csv").exists()
    assert (base / "images" / "13_subgroup_performance.png").exists()
    assert any((base / "images").iterdir())


def test_model_comparison_contains_baseline_and_models(trained_run):
    result, _ = trained_run
    comparison = result["comparison"]

    assert "Dummy (majority class)" in comparison["model"].tolist()
    assert "Logistic Regression" in comparison["model"].tolist()
    assert "Decision Tree" in comparison["model"].tolist()
    assert comparison["validation_pr_auc"].between(0, 1).all()
    assert comparison.iloc[0]["selection_rank"] == 1
    assert {"eligible", "selection_rank", "policy_satisfied"} <= set(comparison.columns)
    assert set(result["selection_policy"]["eligible_models"]) == set(
        comparison.loc[comparison["eligible"], "model"]
    )


def test_predictors_exclude_id_and_target(trained_run):
    result, _ = trained_run

    features = result["feature_lists"]["all"]
    assert ID_COLUMN not in features
    assert "default" not in features
    assert ID_COLUMN not in result["feature_lists"]["numeric"]
    assert "default" not in result["feature_lists"]["numeric"]


def test_threshold_is_selected_from_training_only_out_of_fold_predictions(trained_run):
    result, _ = trained_run
    table = result["threshold_table"]
    selected = result["threshold"]
    train_rows = len(result["splits"]["train"])
    validation_rows = len(result["splits"]["validation"])

    assert result["threshold_selection"]["method"] == "out-of-fold predictions"
    assert result["threshold_selection"]["data"] == "training split only"
    assert result["threshold_selection"]["n_samples"] == train_rows
    assert result["threshold_selection"]["n_samples"] != train_rows + validation_rows
    assert result["threshold_selection"]["model_selection_data"] == "validation split"
    assert len(result["oof_probabilities"]) == train_rows
    assert len(set(result["oof_folds"])) == result["threshold_selection"]["folds"]
    assert table["n_samples"].eq(train_rows).all()
    assert selected in table["threshold"].tolist()
    selected_row = table.loc[table["selected"]].iloc[0]
    assert selected_row["f1"] == pytest.approx(table["f1"].max())
    assert 0 <= result["diagnostic_validation_threshold"] <= 1
    assert result["threshold_selection"]["diagnostic_validation_f1_threshold"] == (
        pytest.approx(result["diagnostic_validation_threshold"])
    )


def test_fairness_analysis_covers_each_attribute_and_flags_small_groups(trained_run):
    result, base = trained_run
    table = result["subgroup_table"]
    test_rows = len(result["splits"]["test"])

    assert {"SEX", "AGE_GROUP", "EDUCATION", "MARRIAGE"} <= set(
        table["group_attribute"]
    )
    for attribute, group in table.groupby("group_attribute"):
        assert group["n_records"].sum() == test_rows, attribute
        assert group["n_defaults"].sum() == int(
            result["splits"]["test"]["default"].sum()
        ), attribute

    report = result["fairness_report"]
    assert report["not_a_fairness_audit"] is True
    assert report["n_small_groups"] == int(table["small_sample"].sum())
    assert report["test_samples"] == test_rows

    with (base / "reports" / "fairness_analysis.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["not_a_fairness_audit"] is True
    assert payload["test_samples"] == test_rows


def test_permutation_importance_is_reported(trained_run):
    result, base = trained_run
    importance = result["feature_importance"]

    assert {"feature", "importance", "importance_std", "method"} <= set(
        importance.columns
    )
    assert importance["method"].str.startswith("permutation").all()
    assert len(importance) == len(result["feature_lists"]["all"])
    assert importance["importance"].is_monotonic_decreasing
    assert np.isfinite(importance["importance"]).all()
    assert result["model_based_importance"] is not None
    assert (base / "reports" / "feature_importance_model_based.csv").exists()


def test_selection_policy_is_documented(trained_run):
    result, base = trained_run
    policy = result["selection_policy"]

    assert policy["recall_floor"] == pytest.approx(0.55)
    assert policy["precision_floor"] == pytest.approx(0.50)
    assert policy["primary_metric"] == "validation_pr_auc"
    assert "financial cost" in policy["cost_caveat"]
    assert result["selected_model"] == policy["ranked_models"][0]["model"]

    with (base / "reports" / "selection_policy.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "cost_sensitivity" in payload
    assert payload["selected_model"] == result["selected_model"]


def test_final_test_metrics_are_reported_honestly(trained_run):
    result, base = trained_run
    metrics = result["final_test_metrics"]
    test_size = len(result["splits"]["test"])

    assert metrics["n_samples"] == test_size
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert 0 <= metrics[key] <= 1
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == test_size

    with (base / "reports" / "final_test_metrics.json").open(
        encoding="utf-8"
    ) as handle:
        payload = json.load(handle)
    assert payload["selected_model"] == result["selected_model"]
    assert payload["threshold"] == pytest.approx(result["threshold"])


def test_saved_model_loads_and_predicts(trained_run, synthetic_raw_df):
    result, _ = trained_run
    artifact = load_model_artifact(result["model_path"])
    records = synthetic_raw_df.drop(columns=["default"]).head(10)

    predictions = predict_customers(records, artifact)

    assert list(predictions.columns) == [ID_COLUMN, "Default Probability", "Prediction"]
    assert len(predictions) == len(records)
    assert predictions["Default Probability"].between(0, 1).all()
    assert set(predictions["Prediction"]) <= {"Default", "No Default"}

    expected = np.where(
        predictions["Default Probability"] >= artifact["threshold"],
        "Default",
        "No Default",
    )
    assert predictions["Prediction"].tolist() == expected.tolist()
    assert predictions[ID_COLUMN].tolist() == records[ID_COLUMN].tolist()


def test_prediction_works_without_target_and_id(trained_run, synthetic_raw_df):
    result, _ = trained_run
    artifact = load_model_artifact(result["model_path"])
    records = synthetic_raw_df.drop(columns=["default", ID_COLUMN]).head(3)

    predictions = predict_customers(records, artifact)

    assert predictions[ID_COLUMN].tolist() == ["Customer 1", "Customer 2", "Customer 3"]


def test_prediction_rejects_missing_required_column(trained_run, synthetic_raw_df):
    result, _ = trained_run
    artifact = load_model_artifact(result["model_path"])
    records = synthetic_raw_df.drop(columns=["default", "LIMIT_BAL"])

    with pytest.raises(PredictionError, match="missing required columns"):
        predict_customers(records, artifact)


def test_prediction_rejects_empty_input(trained_run, synthetic_raw_df):
    result, _ = trained_run
    artifact = load_model_artifact(result["model_path"])
    records = synthetic_raw_df.drop(columns=["default"]).iloc[0:0]

    with pytest.raises(PredictionError, match="no customer records"):
        predict_customers(records, artifact)


def test_prediction_rejects_invalid_category(trained_run, synthetic_raw_df):
    result, _ = trained_run
    artifact = load_model_artifact(result["model_path"])
    records = synthetic_raw_df.drop(columns=["default"]).head(2).copy()
    records.loc[0, "SEX"] = 9

    with pytest.raises(PredictionError, match="Invalid customer record"):
        predict_customers(records, artifact)


def test_prediction_rejects_infinite_values(trained_run, synthetic_raw_df):
    result, _ = trained_run
    artifact = load_model_artifact(result["model_path"])
    records = synthetic_raw_df.drop(columns=["default"]).head(2).copy()
    records["BILL_AMT1"] = records["BILL_AMT1"].astype(float)
    records.loc[0, "BILL_AMT1"] = np.inf

    with pytest.raises(PredictionError, match="Invalid customer record"):
        predict_customers(records, artifact)


def test_resampling_is_not_applied_to_original_training_data(
    synthetic_engineered_df: pd.DataFrame,
):
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    from sklearn.linear_model import LogisticRegression

    from src.preprocessing import build_preprocessor, get_feature_lists, split_dataset

    train, _, _ = split_dataset(synthetic_engineered_df)
    numeric, categorical = get_feature_lists(train)
    features = [column for column in train.columns if column not in {"ID", "default"}]

    class_counts_before = train["default"].value_counts().sort_index()
    features_before = train[features].copy(deep=True)

    pipeline = ImbalancedPipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("sampler", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(train[features], train["default"])

    pd.testing.assert_series_equal(
        train["default"].value_counts().sort_index(), class_counts_before
    )
    pd.testing.assert_frame_equal(train[features], features_before)
    assert hasattr(pipeline.named_steps["sampler"], "sampling_strategy_")

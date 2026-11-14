"""Tests for metrics, threshold behaviour, and evaluation figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    classification_report_frame,
    compute_metrics,
    confusion_counts,
    get_feature_importances,
    out_of_fold_probabilities,
    permutation_importance_table,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_threshold_analysis,
    positive_class_probabilities,
    select_threshold,
    threshold_analysis_table,
)
from src.preprocessing import (
    build_model_pipeline,
    get_feature_lists,
    split_dataset,
)


def test_compute_metrics_returns_expected_values():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.6, 0.9])

    metrics = compute_metrics(y_true, probabilities, threshold=0.5)

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert 0 <= metrics["pr_auc"] <= 1
    assert metrics["predicted_positive_rate"] == pytest.approx(0.75)


def test_compute_metrics_validates_inputs():
    with pytest.raises(ValueError, match="threshold"):
        compute_metrics([0, 1], [0.2, 0.8], threshold=1.5)
    with pytest.raises(ValueError, match="same length"):
        compute_metrics([0, 1], [0.2])
    with pytest.raises(ValueError, match="at least one record"):
        compute_metrics([], [])
    with pytest.raises(ValueError, match="labels 0 and 1"):
        compute_metrics([0, 2], [0.2, 0.8])
    with pytest.raises(ValueError, match="labels 0 and 1"):
        compute_metrics([0, np.nan], [0.2, 0.8])
    with pytest.raises(ValueError, match="finite"):
        compute_metrics([0, 1], [0.2, np.nan])
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_metrics([0, 1], [0.2, 1.2])


class _ReversedClassEstimator:
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        rows = len(X)
        positive = np.linspace(0.1, 0.9, rows)
        return np.column_stack([positive, 1 - positive])


def test_positive_probability_uses_classes_order_not_column_index():
    X = pd.DataFrame({"value": [1, 2, 3]})

    probabilities = positive_class_probabilities(_ReversedClassEstimator(), X)

    assert np.allclose(probabilities, np.linspace(0.1, 0.9, 3))


def test_positive_probability_requires_class_one():
    class _NoPositiveEstimator:
        classes_ = np.array([0, 2])

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    with pytest.raises(ValueError, match="Positive class 1"):
        positive_class_probabilities(_NoPositiveEstimator(), [1, 2])


def test_positive_probability_rejects_malformed_output():
    class _MalformedEstimator:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="unexpected shape"):
        positive_class_probabilities(_MalformedEstimator(), [1, 2])


def test_oof_splits_are_validated():
    X = pd.DataFrame({"value": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])

    with pytest.raises(ValueError, match="overlap"):
        out_of_fold_probabilities(None, X, y, splits=[([0, 1], [1, 2]), ([0, 1], [3])])
    with pytest.raises(ValueError, match="outside"):
        out_of_fold_probabilities(None, X, y, splits=[([0, 1], [2, 4]), ([0, 1], [3])])
    with pytest.raises(ValueError, match="duplicated within"):
        out_of_fold_probabilities(None, X, y, splits=[([0, 1], [2, 2]), ([0, 1], [3])])
    with pytest.raises(ValueError, match="without exactly one"):
        out_of_fold_probabilities(
            None, X, y, splits=[([1, 2, 3], [0]), ([0, 2, 3], [1])]
        )


def test_single_class_has_undefined_ranking_metrics():
    metrics = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])


def test_lower_threshold_increases_recall():
    y_true = np.array([0, 1, 1, 0, 1])
    probabilities = np.array([0.2, 0.3, 0.8, 0.9, 0.55])

    low = compute_metrics(y_true, probabilities, threshold=0.3)
    high = compute_metrics(y_true, probabilities, threshold=0.8)

    assert low["recall"] >= high["recall"]
    assert low["fp"] >= high["fp"]


def test_threshold_table_contains_documented_thresholds():
    table = threshold_analysis_table(
        [0, 1, 1, 0, 1, 0],
        [0.1, 0.9, 0.6, 0.3, 0.7, 0.2],
    )

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        assert threshold in table["threshold"].tolist()
    assert {"precision", "recall", "f1", "tn", "fp", "fn", "tp"} <= set(table.columns)


def test_select_threshold_maximizes_requested_metric():
    y_true = np.array([0] * 10 + [1] * 10)
    probabilities = np.array([0.1] * 10 + [0.9] * 10)

    selected = select_threshold(y_true, probabilities, metric="f1")
    table = threshold_analysis_table(y_true, probabilities)

    assert selected in table["threshold"].tolist()
    assert table.loc[table["threshold"] == selected, "f1"].iloc[0] == pytest.approx(
        table["f1"].max()
    )


def test_select_threshold_is_deterministic():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    probabilities = np.array([0.2, 0.7, 0.6, 0.9, 0.4, 0.1, 0.8, 0.3, 0.55, 0.25])

    first = select_threshold(y_true, probabilities)
    second = select_threshold(y_true, probabilities)

    assert first == second


def test_confusion_counts_and_report():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    counts = confusion_counts(y_true, y_pred)
    report = classification_report_frame(y_true, y_pred)

    assert counts == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert {"class", "precision", "recall", "f1-score", "support"} <= set(
        report.columns
    )
    assert {"No default", "Default"} <= set(report["class"])


def test_feature_importances_are_sorted(synthetic_engineered_df: pd.DataFrame):
    train, _, _ = split_dataset(synthetic_engineered_df)
    numeric, categorical = get_feature_lists(train)
    features = [column for column in train.columns if column not in {"ID", "default"}]

    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42), numeric, categorical
    )
    pipeline.fit(train[features], train["default"])

    importance = get_feature_importances(pipeline, top_n=10)

    assert importance is not None
    assert len(importance) == 10
    assert importance["importance"].is_monotonic_decreasing
    assert positive_class_probabilities(pipeline, train[features]).shape == (
        len(train),
    )


def test_plots_are_written(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.6, 0.9, 0.3, 0.7, 0.2, 0.85])
    predictions = (probabilities >= 0.5).astype(int)

    confusion_path = tmp_path / "confusion.png"
    roc_path = tmp_path / "roc.png"
    pr_path = tmp_path / "pr.png"
    threshold_path = tmp_path / "threshold.png"
    comparison_path = tmp_path / "comparison.png"
    importance_path = tmp_path / "importance.png"

    plot_confusion_matrix(y_true, predictions, confusion_path)
    plot_roc_curves(y_true, {"Model A": probabilities}, roc_path)
    plot_precision_recall_curves(y_true, {"Model A": probabilities}, pr_path)
    table = threshold_analysis_table(y_true, probabilities)
    plot_threshold_analysis(table, 0.5, threshold_path)
    comparison = pd.DataFrame(
        {
            "model": ["Model A", "Model B"],
            "pr_auc": [0.7, 0.6],
            "roc_auc": [0.8, 0.75],
            "f1": [0.6, 0.55],
            "recall": [0.7, 0.6],
            "precision": [0.5, 0.45],
        }
    )
    plot_model_comparison(comparison, comparison_path)
    plot_feature_importance(
        pd.DataFrame({"feature": ["A", "B"], "importance": [0.7, 0.3]}),
        importance_path,
    )

    for path in [
        confusion_path,
        roc_path,
        pr_path,
        threshold_path,
        comparison_path,
        importance_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 1000


def _holdout_pipeline(synthetic_engineered_df: pd.DataFrame):
    train, validation, _ = split_dataset(synthetic_engineered_df)
    numeric, categorical = get_feature_lists(train)
    features = [column for column in train.columns if column not in {"ID", "default"}]
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42), numeric, categorical
    )
    pipeline.fit(train[features], train["default"])
    return pipeline, train, validation, features


def test_out_of_fold_probabilities_cover_every_record(
    synthetic_engineered_df: pd.DataFrame,
):
    pipeline, train, validation, features = _holdout_pipeline(synthetic_engineered_df)
    combined = pd.concat([train, validation], ignore_index=True)

    probabilities, folds = out_of_fold_probabilities(
        pipeline,
        combined[features],
        combined["default"],
        n_splits=4,
        random_state=42,
    )

    assert len(probabilities) == len(combined)
    assert np.isfinite(probabilities).all()
    assert set(folds) == {0, 1, 2, 3}
    assert len(folds) == len(combined)


def test_out_of_fold_predictions_ignore_held_out_labels(
    synthetic_engineered_df: pd.DataFrame,
):
    pipeline, train, validation, features = _holdout_pipeline(synthetic_engineered_df)
    combined = pd.concat([train, validation], ignore_index=True)
    target = combined["default"].copy()

    from sklearn.model_selection import StratifiedKFold

    splits = list(
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(
            combined[features], target
        )
    )
    first, folds = out_of_fold_probabilities(
        pipeline, combined[features], target, splits=splits
    )

    flipped = target.copy()
    held_out = folds == 0
    flipped.iloc[held_out] = 1 - flipped.iloc[held_out]
    second, second_folds = out_of_fold_probabilities(
        pipeline, combined[features], flipped, splits=splits
    )

    np.testing.assert_array_equal(folds, second_folds)
    # Fold 0 predictions come from a model that never saw fold 0 labels.
    np.testing.assert_allclose(first[held_out], second[held_out], atol=1e-12)
    # Other folds train on fold 0, so their predictions must react to the flip.
    assert not np.allclose(first[~held_out], second[~held_out])


def test_permutation_importance_uses_holdout_data(
    synthetic_engineered_df: pd.DataFrame,
):
    pipeline, _, validation, features = _holdout_pipeline(synthetic_engineered_df)

    importance = permutation_importance_table(
        pipeline,
        validation[features],
        validation["default"],
        n_repeats=3,
        random_state=42,
    )

    assert len(importance) == len(features)
    assert {"feature", "importance", "importance_std", "method"} <= set(
        importance.columns
    )
    assert importance["method"].str.startswith("permutation").all()
    assert importance["importance"].is_monotonic_decreasing
    assert np.isfinite(importance["importance"]).all()
    assert np.isfinite(importance["importance_std"]).all()

"""Tests for subgroup metrics, small-sample handling, and fairness reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness import (
    fairness_report,
    plot_subgroup_performance,
    subgroup_performance,
)


@pytest.fixture()
def demographic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SEX": [1, 1, 1, 1, 2, 2, 2, 2],
            "AGE": [25, 35, 45, 55, 25, 35, 45, 55],
            "EDUCATION": [1, 2, 3, 4, 1, 2, 3, 4],
            "MARRIAGE": [1, 2, 3, 1, 2, 3, 1, 2],
        }
    )


def test_subgroup_metrics_are_correct(demographic_frame: pd.DataFrame):
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8])

    table = subgroup_performance(
        demographic_frame, y_true, probabilities, threshold=0.5
    )
    sex_one = table[
        (table["group_attribute"] == "SEX") & (table["group_code"] == "1")
    ].iloc[0]

    assert sex_one["n_records"] == 4
    assert sex_one["n_defaults"] == 2
    assert sex_one["default_rate"] == pytest.approx(0.5)
    assert sex_one["tp"] == 2
    assert sex_one["tn"] == 2
    assert sex_one["fp"] == 0
    assert sex_one["fn"] == 0
    assert sex_one["precision"] == pytest.approx(1.0)
    assert sex_one["recall"] == pytest.approx(1.0)
    assert sex_one["fpr"] == pytest.approx(0.0)
    assert sex_one["fnr"] == pytest.approx(0.0)
    assert bool(sex_one["small_sample"]) is True


def test_each_attribute_partitions_the_records(demographic_frame: pd.DataFrame):
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.full(8, 0.6)

    table = subgroup_performance(
        demographic_frame, y_true, probabilities, threshold=0.5
    )

    for attribute, group in table.groupby("group_attribute"):
        assert group["n_records"].sum() == len(demographic_frame), attribute
        assert group["n_defaults"].sum() == int(y_true.sum()), attribute


def test_groups_without_positives_have_undefined_recall(
    demographic_frame: pd.DataFrame,
):
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])

    table = subgroup_performance(
        demographic_frame, y_true, probabilities, threshold=0.5
    )
    sex_two = table[
        (table["group_attribute"] == "SEX") & (table["group_code"] == "2")
    ].iloc[0]

    assert np.isnan(sex_two["recall"])
    assert np.isnan(sex_two["precision"])
    assert np.isnan(sex_two["f1"])
    assert sex_two["fpr"] == pytest.approx(0.0)
    assert bool(sex_two["precision_defined"]) is False
    assert bool(sex_two["recall_defined"]) is False
    assert bool(sex_two["f1_defined"]) is False
    assert bool(sex_two["fnr_defined"]) is False
    assert bool(sex_two["small_sample"]) is True


def test_fairness_report_flags_small_groups(demographic_frame: pd.DataFrame):
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8])
    table = subgroup_performance(
        demographic_frame, y_true, probabilities, threshold=0.5
    )

    report = fairness_report(
        table,
        model_name="Test model",
        threshold=0.5,
        test_samples=len(demographic_frame),
    )

    assert report["not_a_fairness_audit"] is True
    assert report["n_small_groups"] == len(table)
    assert report["n_groups"] == len(table)
    assert len(report["caveats"]) >= 4
    assert any(
        "not a complete fairness audit" in caveat for caveat in report["caveats"]
    )
    assert report["spread_reliable_groups"]["SEX"]["reliable_groups"] == 0


def test_subgroup_plot_is_written(demographic_frame: pd.DataFrame, tmp_path):
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8])
    table = subgroup_performance(
        demographic_frame, y_true, probabilities, threshold=0.5
    )

    path = plot_subgroup_performance(table, tmp_path / "fairness.png", threshold=0.5)

    assert path.exists()
    assert path.stat().st_size > 1000


def test_subgroup_performance_validates_alignment(demographic_frame: pd.DataFrame):
    with pytest.raises(ValueError, match="same records"):
        subgroup_performance(
            demographic_frame, np.array([0, 1]), np.array([0.2, 0.8]), threshold=0.5
        )


def test_subgroup_performance_validates_inputs(demographic_frame: pd.DataFrame):
    y_true = np.zeros(8, dtype=int)
    probabilities = np.full(8, 0.5)

    with pytest.raises(ValueError, match="threshold"):
        subgroup_performance(demographic_frame, y_true, probabilities, threshold=1.5)
    with pytest.raises(ValueError, match="binary labels"):
        subgroup_performance(
            demographic_frame, np.full(8, 2), probabilities, threshold=0.5
        )
    with pytest.raises(ValueError, match="finite"):
        subgroup_performance(
            demographic_frame, y_true, np.full(8, np.nan), threshold=0.5
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        subgroup_performance(demographic_frame, y_true, np.full(8, 1.5), threshold=0.5)
    with pytest.raises(TypeError, match="DataFrame"):
        subgroup_performance(
            [1, 2], np.array([0, 1]), np.array([0.2, 0.8]), threshold=0.5
        )


def test_age_groups_do_not_mislabel_young_customers(demographic_frame: pd.DataFrame):
    frame = demographic_frame.copy()
    frame.loc[0, "AGE"] = 18
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.full(8, 0.6)

    table = subgroup_performance(
        frame, y_true, probabilities, threshold=0.5, attributes=("AGE_GROUP",)
    )
    young = table[table["group"] == "29 or younger"].iloc[0]

    assert young["n_records"] == 2
    assert "21-29" not in table["group"].tolist()


def test_missing_demographics_form_an_explicit_group(demographic_frame: pd.DataFrame):
    frame = demographic_frame.copy()
    frame.loc[[0, 1], "SEX"] = np.nan
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.full(8, 0.6)

    table = subgroup_performance(frame, y_true, probabilities, threshold=0.5)
    sex_rows = table[table["group_attribute"] == "SEX"]

    assert sex_rows["n_records"].sum() == len(frame)
    missing = sex_rows[sex_rows["group_code"] == "missing"].iloc[0]
    assert missing["group"] == "Missing"
    assert missing["n_records"] == 2
    assert bool(missing["small_sample"]) is True


def test_fairness_report_counts_undefined_and_missing_groups(
    demographic_frame: pd.DataFrame,
):
    frame = demographic_frame.copy()
    frame.loc[0, "SEX"] = np.nan
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])
    table = subgroup_performance(frame, y_true, probabilities, threshold=0.5)

    report = fairness_report(
        table, model_name="Test model", threshold=0.5, test_samples=len(frame)
    )

    assert report["n_groups_with_undefined_precision"] >= 1
    assert report["n_groups_with_undefined_recall"] >= 1
    assert any(
        row["group_attribute"] == "SEX" for row in report["missing_group_records"]
    )
    assert any(
        row["group_attribute"] == "SEX" for row in report["undefined_metric_groups"]
    )


def test_subgroup_plot_supports_any_attribute_count(
    demographic_frame: pd.DataFrame, tmp_path
):
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.full(8, 0.6)
    table = subgroup_performance(
        demographic_frame, y_true, probabilities, threshold=0.5
    )
    extra = table[table["group_attribute"] == "SEX"].copy()
    extra["group_attribute"] = "REGION"
    extended = pd.concat([table, extra], ignore_index=True)

    path = plot_subgroup_performance(
        extended,
        tmp_path / "fairness.png",
        threshold=0.5,
        attributes=("SEX", "AGE_GROUP", "EDUCATION", "MARRIAGE", "REGION"),
    )

    assert path.exists()
    assert path.stat().st_size > 1000


def test_fairness_report_rejects_empty_table():
    with pytest.raises(ValueError, match="empty"):
        fairness_report(
            pd.DataFrame(), model_name="Test model", threshold=0.5, test_samples=0
        )

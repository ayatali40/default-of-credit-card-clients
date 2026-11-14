"""Tests for the documented model-selection policy."""

from __future__ import annotations

import pandas as pd
import pytest

from src.selection import (
    apply_selection_policy,
    expected_cost_table,
    model_selection_policy,
)


def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["Model A", "Model B", "Model C", "Dummy"],
            "role": ["base", "imbalance", "tuned", "baseline"],
            "validation_precision": [0.60, 0.45, 0.70, 0.00],
            "validation_recall": [0.60, 0.70, 0.30, 0.00],
            "validation_f1": [0.60, 0.55, 0.42, 0.00],
            "validation_roc_auc": [0.80, 0.79, 0.81, 0.50],
            "validation_pr_auc": [0.55, 0.60, 0.65, 0.22],
            "validation_tn": [80, 70, 90, 90],
            "validation_fp": [40, 60, 30, 100],
            "validation_fn": [40, 30, 70, 100],
            "validation_tp": [60, 70, 30, 0],
        }
    )


def test_policy_prefers_pr_auc_among_eligible_models():
    ranked = apply_selection_policy(_comparison())

    assert ranked.iloc[0]["model"] == "Model A"
    assert bool(ranked.iloc[0]["eligible"]) is True
    assert ranked.iloc[0]["selection_rank"] == 1
    assert bool(ranked.iloc[0]["policy_satisfied"]) is True
    # B fails the precision floor; C fails the recall floor; the baseline is never eligible.
    assert ranked.loc[ranked["model"] == "Model B", "eligible"].iloc[0] == False
    assert ranked.loc[ranked["model"] == "Model C", "eligible"].iloc[0] == False
    assert ranked.loc[ranked["model"] == "Dummy", "eligible"].iloc[0] == False
    # Ineligible models are still ranked by PR-AUC after eligible ones.
    assert ranked["model"].tolist() == ["Model A", "Model C", "Model B", "Dummy"]


def test_policy_falls_back_to_highest_pr_auc():
    comparison = _comparison()
    comparison.loc[0, "validation_precision"] = 0.40

    ranked = apply_selection_policy(comparison)

    assert not ranked["eligible"].any()
    assert ranked.iloc[0]["model"] == "Model C"  # highest PR-AUC (0.65)
    assert bool(ranked.iloc[0]["policy_satisfied"]) is False


def test_policy_accepts_custom_floors():
    ranked = apply_selection_policy(
        _comparison(), recall_floor=0.65, precision_floor=0.40
    )

    assert ranked.iloc[0]["model"] == "Model B"


def test_baseline_cannot_win_through_fallback_ranking():
    comparison = _comparison()
    comparison.loc[0, "validation_precision"] = 0.40
    comparison.loc[1, "validation_precision"] = 0.40
    comparison.loc[2, "validation_precision"] = 0.40
    comparison.loc[0, "validation_recall"] = 0.30
    comparison.loc[1, "validation_recall"] = 0.30
    comparison.loc[2, "validation_recall"] = 0.30

    ranked = apply_selection_policy(comparison)

    assert not ranked["eligible"].any()
    assert ranked.iloc[0]["model"] != "Dummy"
    assert ranked.iloc[-1]["model"] == "Dummy"


def test_empty_comparison_table_is_handled():
    empty = pd.DataFrame(
        {
            "model": pd.Series(dtype=str),
            "role": pd.Series(dtype=str),
            "validation_precision": pd.Series(dtype=float),
            "validation_recall": pd.Series(dtype=float),
            "validation_f1": pd.Series(dtype=float),
            "validation_roc_auc": pd.Series(dtype=float),
            "validation_pr_auc": pd.Series(dtype=float),
        }
    )

    ranked = apply_selection_policy(empty)

    assert ranked.empty
    assert {"eligible", "selection_rank", "policy_satisfied"} <= set(ranked.columns)


def test_cost_sensitivity_is_reported_for_every_ratio():
    table = expected_cost_table(_comparison(), ratios=(1, 5, 10), threshold=0.5)

    assert set(table["fn_fp_cost_ratio"]) == {1, 5, 10}
    assert len(table) == 9
    assert "Dummy" not in table["model"].tolist()
    assert (table["reference_threshold"] == 0.5).all()
    assert table["hypothetical_fn_fp_ratio"].all()
    for _, group in table.groupby("model"):
        assert group.sort_values("fn_fp_cost_ratio")[
            "expected_cost_per_1000_scored"
        ].is_monotonic_increasing


def test_policy_documents_the_missing_cost_matrix():
    policy = model_selection_policy()

    assert policy["recall_floor"] == pytest.approx(0.55)
    assert policy["precision_floor"] == pytest.approx(0.50)
    assert policy["primary_metric"] == "validation_pr_auc"
    assert "modelling objective" in policy["cost_caveat"]
    assert "financial cost" in policy["cost_caveat"]
    assert "reference threshold" in policy["eligibility_caveat"]
    assert policy["baseline_excluded_from_ranking"] is True
    assert policy["selection_data"] == "validation split"
    assert "training split only" in policy["threshold_calibration_data"]

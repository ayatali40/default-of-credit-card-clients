"""Model selection policy for imbalanced credit default prediction."""

from __future__ import annotations

import pandas as pd

REFERENCE_THRESHOLD = 0.50
RECALL_FLOOR = 0.55
PRECISION_FLOOR = 0.50
PRIMARY_METRIC = "validation_pr_auc"
TIE_BREAKERS = ("validation_f1", "validation_roc_auc")
COST_RATIOS = (1, 5, 10)


def model_selection_policy() -> dict:
    return {
        "reference_threshold": REFERENCE_THRESHOLD,
        "recall_floor": RECALL_FLOOR,
        "precision_floor": PRECISION_FLOOR,
        "primary_metric": PRIMARY_METRIC,
        "tie_breakers": list(TIE_BREAKERS),
        "rule": (
            f"At threshold {REFERENCE_THRESHOLD}, require recall >= {RECALL_FLOOR} "
            f"and precision >= {PRECISION_FLOOR}; pick highest {PRIMARY_METRIC}."
        ),
        "fallback": "If no model meets floors, pick highest PR-AUC among non-baseline.",
        "baseline_excluded_from_ranking": True,
    }


def apply_selection_policy(
    comparison: pd.DataFrame,
    recall_floor: float = RECALL_FLOOR,
    precision_floor: float = PRECISION_FLOOR,
    primary_metric: str = PRIMARY_METRIC,
    tie_breakers: tuple[str, ...] = TIE_BREAKERS,
) -> pd.DataFrame:
    required = {
        "model",
        "role",
        "validation_precision",
        "validation_recall",
        primary_metric,
        *tie_breakers,
    }
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"Comparison table is missing columns: {missing}")

    frame = comparison.copy()
    if frame.empty:
        for col in [
            "meets_recall_floor",
            "meets_precision_floor",
            "is_candidate",
            "eligible",
            "selection_rank",
            "policy_satisfied",
        ]:
            frame[col] = pd.Series(dtype=bool if col != "selection_rank" else int)
        return frame

    frame["is_candidate"] = frame["role"] != "baseline"
    frame["meets_recall_floor"] = frame["validation_recall"] >= recall_floor
    frame["meets_precision_floor"] = frame["validation_precision"] >= precision_floor
    frame["eligible"] = (
        frame["meets_recall_floor"]
        & frame["meets_precision_floor"]
        & frame["is_candidate"]
    )

    sort_columns = [
        "is_candidate",
        "eligible",
        primary_metric,
        *tie_breakers,
        "model",
    ]
    ascending = [False, False, *([False] * (1 + len(tie_breakers))), True]
    frame = frame.sort_values(
        sort_columns, ascending=ascending, kind="mergesort"
    ).reset_index(drop=True)
    frame["selection_rank"] = frame.index + 1
    frame["policy_satisfied"] = bool(frame.iloc[0]["eligible"])
    return frame


def expected_cost_table(
    comparison: pd.DataFrame,
    ratios: tuple[int, ...] = COST_RATIOS,
    per_customers: int = 1000,
    threshold: float = REFERENCE_THRESHOLD,
) -> pd.DataFrame:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie between 0 and 1.")

    required = {
        "model",
        "role",
        "validation_fn",
        "validation_fp",
        "validation_tn",
        "validation_tp",
    }
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"Comparison table is missing columns: {missing}")

    rows = []
    for _, row in comparison.iterrows():
        if row["role"] == "baseline":
            continue
        n_scored = int(
            row["validation_fn"]
            + row["validation_fp"]
            + row["validation_tn"]
            + row["validation_tp"]
        )
        for ratio in ratios:
            cost = (ratio * row["validation_fn"] + row["validation_fp"]) / max(
                n_scored, 1
            )
            rows.append(
                {
                    "model": row["model"],
                    "reference_threshold": float(threshold),
                    "fn_fp_cost_ratio": int(ratio),
                    "expected_cost_per_1000_scored": round(
                        float(cost * per_customers), 3
                    ),
                    "validation_fn": int(row["validation_fn"]),
                    "validation_fp": int(row["validation_fp"]),
                    "validation_samples": n_scored,
                }
            )
    return pd.DataFrame(rows)

"""Subgroup performance analysis by demographic groups."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_cleaning import EDUCATION_LABELS, MARRIAGE_LABELS, SEX_LABELS
from src.evaluation import confusion_counts

GROUP_ATTRIBUTES = ("SEX", "AGE_GROUP", "EDUCATION", "MARRIAGE")
METRICS = ("precision", "recall", "f1", "fpr", "fnr")
PLOT_METRICS = ("precision", "recall", "fpr", "fnr")

MIN_GROUP_SIZE = 200
MIN_GROUP_POSITIVES = 30

AGE_BINS = [-np.inf, 29, 39, 49, 59, np.inf]
AGE_LABELS = ["29 or younger", "30-39", "40-49", "50-59", "60+"]
MISSING_GROUP_CODE = "missing"
MISSING_GROUP_LABEL = "Missing"


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    if "AGE" not in df.columns:
        raise ValueError("Cannot build age groups without an AGE column.")
    result = df.copy()
    result["AGE_GROUP"] = pd.cut(
        result["AGE"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
        include_lowest=True,
    )
    return result


def _attribute_groups(series: pd.Series) -> list:
    if isinstance(series.dtype, pd.CategoricalDtype):
        groups = [v for v in series.cat.categories if (series == v).any()]
    else:
        groups = sorted(series.dropna().unique().tolist())
    if series.isna().any():
        groups.append(None)
    return groups


def _group_label(attribute: str, value) -> str:
    if value is None:
        return MISSING_GROUP_LABEL
    mapping = {
        "SEX": SEX_LABELS,
        "EDUCATION": EDUCATION_LABELS,
        "MARRIAGE": MARRIAGE_LABELS,
    }.get(attribute)
    if mapping is None:
        return str(value)
    try:
        return mapping.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def _group_code(value) -> str:
    return MISSING_GROUP_CODE if value is None else str(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _validate_subgroup_inputs(target, scores, threshold) -> None:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be within [0, 1]")


def subgroup_performance(
    df: pd.DataFrame,
    y_true,
    probabilities: np.ndarray,
    threshold: float,
    attributes: tuple[str, ...] = GROUP_ATTRIBUTES,
    min_group_size: int = MIN_GROUP_SIZE,
    min_positives: int = MIN_GROUP_POSITIVES,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    target = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and probabilities must be 1-d")
    if len(df) != len(target) or len(scores) != len(target):
        raise ValueError("df, y_true, and probabilities length mismatch")
    try:
        target = target.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must contain only 0 and 1") from exc
    _validate_subgroup_inputs(target, scores, threshold)

    frame = add_age_group(df) if "AGE_GROUP" in attributes else df.copy()
    predictions = (scores >= threshold).astype(int)

    rows = []
    for attribute in attributes:
        if attribute not in frame.columns:
            raise ValueError(f"Grouping column '{attribute}' is missing")
        series = frame[attribute]
        for value in _attribute_groups(series):
            mask = (
                series.isna().to_numpy()
                if value is None
                else (series == value).to_numpy()
            )
            n_records = int(mask.sum())
            if n_records == 0:
                continue

            counts = confusion_counts(target[mask], predictions[mask])
            tp, fp = counts["tp"], counts["fp"]
            fn, tn = counts["fn"], counts["tn"]
            n_defaults = int(tp + fn)
            precision_defined = (tp + fp) > 0
            recall_defined = n_defaults > 0
            fpr_defined = (fp + tn) > 0

            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)
            if precision_defined and recall_defined and precision + recall:
                f1 = float(2 * precision * recall / (precision + recall))
            elif precision_defined and recall_defined:
                f1 = 0.0
            else:
                f1 = float("nan")

            rows.append(
                {
                    "group_attribute": attribute,
                    "group": _group_label(attribute, value),
                    "group_code": _group_code(value),
                    "n_records": n_records,
                    "n_defaults": n_defaults,
                    "default_rate": round(_safe_ratio(n_defaults, n_records), 6),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fpr": _safe_ratio(fp, fp + tn),
                    "fnr": _safe_ratio(fn, fn + tp),
                    "precision_defined": bool(precision_defined),
                    "recall_defined": bool(recall_defined),
                    "f1_defined": bool(precision_defined and recall_defined),
                    "fpr_defined": bool(fpr_defined),
                    "fnr_defined": bool(recall_defined),
                    "small_sample": bool(
                        n_records < min_group_size or n_defaults < min_positives
                    ),
                }
            )
    return pd.DataFrame(rows)


def _spread(values: pd.Series) -> float | None:
    clean = values.dropna()
    return float(clean.max() - clean.min()) if len(clean) >= 2 else None


def fairness_report(
    table: pd.DataFrame,
    model_name: str,
    threshold: float,
    test_samples: int,
    min_group_size: int = MIN_GROUP_SIZE,
    min_positives: int = MIN_GROUP_POSITIVES,
) -> dict:
    if table.empty:
        raise ValueError("Empty subgroup table")

    small = table[table["small_sample"]]
    undefined_precision = table[~table["precision_defined"]]
    undefined_recall = table[~table["recall_defined"]]
    missing_groups = table[table["group_code"] == MISSING_GROUP_CODE]
    spreads = {}
    for attribute, group in table.groupby("group_attribute", sort=False):
        reliable = group[~group["small_sample"]]
        spreads[attribute] = {m: _spread(reliable[m]) for m in METRICS}
        spreads[attribute]["reliable_groups"] = len(reliable)

    return {
        "model": model_name,
        "threshold": float(threshold),
        "test_samples": int(test_samples),
        "group_attributes": list(table["group_attribute"].drop_duplicates()),
        "min_group_size": int(min_group_size),
        "min_positives": int(min_positives),
        "n_groups": len(table),
        "n_small_groups": len(small),
        "small_groups": small[
            ["group_attribute", "group", "n_records", "n_defaults"]
        ].to_dict(orient="records"),
        "n_groups_with_undefined_precision": len(undefined_precision),
        "n_groups_with_undefined_recall": len(undefined_recall),
        "undefined_metric_groups": undefined_precision[
            ["group_attribute", "group", "n_records", "n_defaults"]
        ].to_dict(orient="records"),
        "missing_group_records": missing_groups[
            ["group_attribute", "group", "n_records", "n_defaults"]
        ].to_dict(orient="records"),
        "spread_reliable_groups": spreads,
    }


def plot_subgroup_performance(
    table: pd.DataFrame,
    path: Path,
    threshold: float,
    attributes: tuple[str, ...] = GROUP_ATTRIBUTES,
) -> Path:
    if not attributes:
        raise ValueError("At least one grouping attribute required")

    colors = {
        "precision": "#2a6f97",
        "recall": "#d1495b",
        "fpr": "#edae49",
        "fnr": "#6a994e",
    }
    metric_labels = {
        "precision": "Precision",
        "recall": "Recall",
        "fpr": "FPR",
        "fnr": "FNR",
    }

    n_columns = 2 if len(attributes) > 1 else 1
    n_rows = int(np.ceil(len(attributes) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(7 * n_columns, 4.5 * n_rows), squeeze=False
    )
    for ax, attribute in zip(axes.ravel(), attributes):
        group = table[table["group_attribute"] == attribute]
        if group.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(group))
        width = 0.2
        for offset, metric in enumerate(PLOT_METRICS):
            heights = group[metric].to_numpy(dtype=float)
            bars = ax.bar(
                x + (offset - 1.5) * width,
                heights,
                width,
                label=metric_labels[metric],
                color=colors[metric],
            )
            for bar, small in zip(bars, group["small_sample"]):
                if small:
                    bar.set_hatch("//")
                    bar.set_edgecolor("#333333")

        tick_labels = [
            f"{row.group}\nn={row.n_records}" + (" †" if row.small_sample else "")
            for row in group.itertuples()
        ]
        ax.set_xticks(x, tick_labels, fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score")
        ax.set_title(attribute.replace("_", " ").title())
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.ravel()[len(attributes) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Subgroup performance (threshold={threshold:.2f})", fontsize=12
    )
    fig.text(
        0.01,
        0.01,
        f"† = small group (n < {MIN_GROUP_SIZE} or < {MIN_GROUP_POSITIVES} defaults)",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

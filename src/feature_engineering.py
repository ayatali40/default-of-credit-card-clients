"""Feature engineering for credit card default data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import BILL_COLUMNS, PAY_COLUMNS, PAYMENT_COLUMNS

ENGINEERED_FEATURES = [
    "BILL_AVG",
    "BILL_TOTAL",
    "PAY_AVG",
    "PAY_TOTAL",
    "NUM_DELAYED_MONTHS",
    "NUM_SEVERE_DELAY_MONTHS",
    "MAX_DELAY_STATUS",
    "AVG_DELAY_STATUS",
    "UTILIZATION_AVG",
    "UTILIZATION_LAST",
    "PAYMENT_TO_BILL_RATIO",
    "ZERO_PAYMENT_MONTHS",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.Series(numerator, dtype="float64")
    den = pd.Series(denominator, dtype="float64")
    zero = den == 0
    result = num / den.mask(zero)
    result = result.mask(zero, 0.0)
    return result.replace([np.inf, -np.inf], 0.0)


def engineer_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    required = set(PAY_COLUMNS + BILL_COLUMNS + PAYMENT_COLUMNS + ["LIMIT_BAL"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {missing}")

    result = df if inplace else df.copy()

    bills = result[BILL_COLUMNS].astype("float64")
    payments = result[PAYMENT_COLUMNS].astype("float64")
    statuses = result[PAY_COLUMNS].astype("float64")

    result["BILL_AVG"] = bills.mean(axis=1, skipna=False)
    result["BILL_TOTAL"] = bills.sum(axis=1, skipna=False)
    result["PAY_AVG"] = payments.mean(axis=1, skipna=False)
    result["PAY_TOTAL"] = payments.sum(axis=1, skipna=False)

    statuses_complete = statuses.notna().all(axis=1)
    result["NUM_DELAYED_MONTHS"] = (
        (statuses > 0).sum(axis=1).astype("float64").where(statuses_complete)
    )
    result["NUM_SEVERE_DELAY_MONTHS"] = (
        (statuses > 1).sum(axis=1).astype("float64").where(statuses_complete)
    )
    result["MAX_DELAY_STATUS"] = statuses.max(axis=1, skipna=False)
    result["AVG_DELAY_STATUS"] = statuses.mean(axis=1, skipna=False)

    result["UTILIZATION_AVG"] = safe_divide(result["BILL_AVG"], result["LIMIT_BAL"])
    result["UTILIZATION_LAST"] = safe_divide(result["BILL_AMT1"], result["LIMIT_BAL"])
    result["PAYMENT_TO_BILL_RATIO"] = safe_divide(
        result["PAY_TOTAL"], result["BILL_TOTAL"].abs()
    )
    result["ZERO_PAYMENT_MONTHS"] = (
        (payments == 0)
        .sum(axis=1)
        .astype("float64")
        .where(payments.notna().all(axis=1))
    )

    return result

"""Shared pytest fixtures built on small synthetic datasets.

The tests never download the real dataset; the fixtures reproduce its schema
and its awkward category codes so the cleaning and leakage rules can be
verified quickly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_synthetic_dataframe(n: int = 360, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic dataset with the official column schema."""
    rng = np.random.default_rng(seed)

    limit_bal = rng.choice([0, 10_000, 30_000, 80_000, 200_000, 500_000], size=n)
    sex = rng.choice([1, 2], size=n)
    education = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n)
    marriage = rng.choice([0, 1, 2, 3], size=n)
    age = rng.integers(21, 70, size=n)
    pay_status = rng.integers(-2, 9, size=(n, 6))
    bills = rng.integers(-2_000, 100_000, size=(n, 6))
    payments = rng.integers(0, 30_000, size=(n, 6))

    risk = (
        0.55 * np.clip(pay_status.max(axis=1), 0, None)
        + 3.0 * (bills.mean(axis=1) / np.maximum(limit_bal, 1))
        - 0.4 * np.log1p(payments.sum(axis=1))
    )
    risky = risk + rng.normal(0, 1.0, size=n)
    target = np.zeros(n, dtype=int)
    target[np.argsort(risky)[-int(0.3 * n) :]] = 1

    data = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
    }
    for index in range(6):
        month = index + 1
        data[f"PAY_{month}"] = pay_status[:, index]
        data[f"BILL_AMT{month}"] = bills[:, index]
        data[f"PAY_AMT{month}"] = payments[:, index]
    data["default"] = target

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def synthetic_raw_df() -> pd.DataFrame:
    """Raw-format synthetic data (canonical column names)."""
    return make_synthetic_dataframe()


@pytest.fixture()
def synthetic_cleaned_df(synthetic_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned synthetic data as a fresh copy."""
    from src.data_cleaning import clean_dataset

    return clean_dataset(synthetic_raw_df.copy())


@pytest.fixture()
def synthetic_engineered_df(synthetic_cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned and feature-engineered synthetic data as a fresh copy."""
    from src.feature_engineering import engineer_features

    return engineer_features(synthetic_cleaned_df.copy())

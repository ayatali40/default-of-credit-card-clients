"""Data cleaning for the credit card default dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import (
    BILL_COLUMNS,
    ID_COLUMN,
    PAY_COLUMNS,
    PAYMENT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    standardize_columns,
)

EDUCATION_LABELS = {
    1: "graduate school",
    2: "university",
    3: "high school",
    4: "others/unknown",
}
MARRIAGE_LABELS = {
    1: "married",
    2: "single",
    3: "others/unknown",
}
SEX_LABELS = {1: "male", 2: "female"}

VALID_EDUCATION = {1, 2, 3, 4}
VALID_MARRIAGE = {1, 2, 3}
VALID_SEX = {1, 2}
VALID_TARGET = {0, 1}

EDUCATION_CODE_MAP = {0: 4, 5: 4, 6: 4}
MARRIAGE_CODE_MAP = {0: 3}

NUMERIC_COLUMNS = ["LIMIT_BAL", "AGE", *PAY_COLUMNS, *BILL_COLUMNS, *PAYMENT_COLUMNS]


def clean_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "EDUCATION" in result:
        result["EDUCATION"] = result["EDUCATION"].replace(EDUCATION_CODE_MAP)
        invalid = set(result["EDUCATION"].dropna().unique()) - VALID_EDUCATION
        if invalid:
            raise ValueError(f"Unexpected EDUCATION codes: {sorted(invalid)}")

    if "MARRIAGE" in result:
        result["MARRIAGE"] = result["MARRIAGE"].replace(MARRIAGE_CODE_MAP)
        invalid = set(result["MARRIAGE"].dropna().unique()) - VALID_MARRIAGE
        if invalid:
            raise ValueError(f"Unexpected MARRIAGE codes: {sorted(invalid)}")

    if "SEX" in result:
        invalid = set(result["SEX"].dropna().unique()) - VALID_SEX
        if invalid:
            raise ValueError(f"Unexpected SEX codes: {sorted(invalid)}")

    return result


def validate_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing")
    values = df[target_column]
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_mask = numeric.isna() | ~numeric.isin(list(VALID_TARGET))
    if invalid_mask.any():
        invalid = values[invalid_mask].unique().tolist()
        raise ValueError(
            f"Target has invalid values (expected 0/1): {invalid}"
        )


def clean_feature_values(df: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
    result = standardize_columns(df) if standardize else df.copy()

    required = [c for c in REQUIRED_COLUMNS if c != TARGET_COLUMN]
    missing = sorted(set(required) - set(result.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(result[col], errors="raise")
        values = converted.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"Column '{col}' has infinite values")
        result[col] = converted

    return clean_categorical_codes(result)


def clean_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    result = standardize_columns(df)

    missing = sorted(set(REQUIRED_COLUMNS) - set(result.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    validate_target(result, target_column)

    if drop_duplicates:
        result = result.drop_duplicates().reset_index(drop=True)

    result = clean_feature_values(result, standardize=False)
    result[target_column] = result[target_column].astype(int)
    return result.reset_index(drop=True)


def _unexpected_code_key(code):
    if isinstance(code, (int, np.integer)):
        return int(code)
    if isinstance(code, (float, np.floating)):
        return int(code) if float(code).is_integer() else float(code)
    return str(code)


def count_unexpected_codes(df: pd.DataFrame) -> dict[str, dict[object, int]]:
    report = {}
    if "EDUCATION" in df:
        counts = df["EDUCATION"].value_counts().sort_index()
        report["EDUCATION"] = {
            _unexpected_code_key(code): int(count)
            for code, count in counts.items()
            if code not in VALID_EDUCATION
        }
    if "MARRIAGE" in df:
        counts = df["MARRIAGE"].value_counts().sort_index()
        report["MARRIAGE"] = {
            _unexpected_code_key(code): int(count)
            for code, count in counts.items()
            if code not in VALID_MARRIAGE
        }
    return report


def data_quality_report(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict:
    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "dtypes": {str(c): str(d) for c, d in df.dtypes.items()},
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {
            str(c): int(n) for c, n in df.isna().sum().items() if n > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": (
            int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df else 0
        ),
        "unexpected_category_codes": count_unexpected_codes(df),
        "unique_values": {
            str(c): int(df[c].nunique())
            for c in ["SEX", "EDUCATION", "MARRIAGE"]
            if c in df
        },
    }
    if target_column in df:
        counts = df[target_column].value_counts().sort_index()
        report["target_distribution"] = {
            str(int(k)): int(v) for k, v in counts.items()
        }
        default_count = int(counts.get(1, 0))
        report["default_rate"] = round(default_count / max(len(df), 1), 6)
        non_default = int(counts.get(0, 0))
        report["imbalance_ratio"] = (
            round(non_default / default_count, 4) if default_count else None
        )
    return report

"""Train/val/test split and preprocessing pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import ID_COLUMN, TARGET_COLUMN

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]


def get_feature_lists(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[list[str], list[str]]:
    excluded = {ID_COLUMN, target_column}
    missing_cat = [c for c in CATEGORICAL_FEATURES if c not in df.columns]
    if missing_cat:
        raise ValueError(f"Missing categorical features: {missing_cat}")

    categorical = list(CATEGORICAL_FEATURES)
    unclassified = [
        c
        for c in df.columns
        if c not in excluded
        and c not in categorical
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if unclassified:
        raise ValueError(
            f"Columns neither numeric nor categorical: {unclassified}"
        )

    numeric = [
        c for c in df.columns if c not in excluded and c not in categorical
    ]
    return numeric, categorical


def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' missing")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be in (0, 1)")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be < 1")

    frame = df.reset_index(drop=True)
    y = frame[target_column]
    train_val, test = train_test_split(
        frame, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=rel_val,
        random_state=random_state,
        stratify=train_val[target_column],
    )

    overlap = (
        set(train.index) & set(validation.index)
        | set(train.index) & set(test.index)
        | set(validation.index) & set(test.index)
    )
    if overlap:
        raise RuntimeError("Split indices overlap")

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise ValueError("Need at least one feature")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    model, numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", model),
        ]
    )


def class_distribution(
    y: pd.Series, target_column: str = TARGET_COLUMN
) -> pd.DataFrame:
    if len(y) == 0:
        raise ValueError("Empty series")
    counts = y.value_counts().sort_index()
    return pd.DataFrame(
        {
            target_column: counts.index.astype(int),
            "count": counts.to_numpy(),
            "proportion": (counts / len(y)).round(4).to_numpy(),
        }
    )

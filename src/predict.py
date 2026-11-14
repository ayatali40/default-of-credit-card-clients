"""Score new customer records with the saved model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_cleaning import clean_feature_values
from src.data_loader import (
    ID_COLUMN,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    standardize_columns,
)
from src.evaluation import positive_class_probabilities
from src.feature_engineering import engineer_features

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
REQUIRED_INPUT_COLUMNS = [
    c for c in REQUIRED_COLUMNS if c not in {ID_COLUMN, TARGET_COLUMN}
]
ARTIFACT_KEYS = ("pipeline", "threshold", "feature_columns")


class PredictionError(ValueError):
    pass


def _validate_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise PredictionError(
            f"Expected dict artefact, got {type(artifact).__name__}"
        )
    missing = [k for k in ARTIFACT_KEYS if k not in artifact]
    if missing:
        raise PredictionError(f"Missing artefact keys: {missing}")

    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise PredictionError("Pipeline has no predict_proba")

    try:
        threshold = float(artifact["threshold"])
    except (TypeError, ValueError) as exc:
        raise PredictionError(
            f"Threshold is not numeric: {artifact['threshold']!r}"
        ) from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise PredictionError(f"Threshold out of range: {threshold}")

    feature_columns = artifact["feature_columns"]
    if (
        not isinstance(feature_columns, (list, tuple))
        or not feature_columns
        or not all(isinstance(c, str) and c for c in feature_columns)
    ):
        raise PredictionError("feature_columns must be a non-empty list of strings")
    if len(set(feature_columns)) != len(feature_columns):
        raise PredictionError("feature_columns contains duplicates")

    return artifact


def load_model_artifact(model_path: Path | str = MODEL_PATH) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Train first with `python -m src.train`."
        )
    try:
        artifact = joblib.load(path)
    except (
        OSError,
        EOFError,
        IndexError,
        pickle.UnpicklingError,
        ValueError,
        KeyError,
        ImportError,
    ) as exc:
        raise PredictionError(f"Could not load model at {path}: {exc}") from exc
    return _validate_artifact(artifact)


def validate_customer_records(records: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise PredictionError("Input must be a DataFrame")
    if records.empty:
        raise PredictionError("No customer records provided")

    try:
        frame = standardize_columns(records)
    except ValueError as exc:
        raise PredictionError(f"Invalid records: {exc}") from exc

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise PredictionError(
            f"Missing columns: {missing}. Required: {REQUIRED_INPUT_COLUMNS}"
        )

    kept = [ID_COLUMN] if ID_COLUMN in frame.columns else []
    kept.extend(REQUIRED_INPUT_COLUMNS)
    frame = frame.loc[:, kept].copy()

    if ID_COLUMN not in frame.columns:
        frame[ID_COLUMN] = [f"Customer {i + 1}" for i in range(len(frame))]
    return frame


def predict_customers(
    records: pd.DataFrame, artifact: dict | None = None
) -> pd.DataFrame:
    model_artifact = (
        load_model_artifact() if artifact is None else _validate_artifact(artifact)
    )

    validated = validate_customer_records(records)
    try:
        cleaned = clean_feature_values(validated)
        engineered = engineer_features(cleaned)
    except ValueError as exc:
        raise PredictionError(f"Invalid record: {exc}") from exc

    feature_columns = list(model_artifact["feature_columns"])
    missing_features = [c for c in feature_columns if c not in engineered.columns]
    if missing_features:
        raise PredictionError(
            f"Feature engineering missing columns: {missing_features}"
        )

    try:
        probs = positive_class_probabilities(
            model_artifact["pipeline"], engineered[feature_columns]
        )
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"Scoring failed: {exc}") from exc

    threshold = float(model_artifact["threshold"])
    return pd.DataFrame(
        {
            ID_COLUMN: engineered[ID_COLUMN].to_numpy(),
            "Default Probability": np.round(probs, 6),
            "Prediction": np.where(
                probs >= threshold, "Default", "No Default"
            ),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict credit card default")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV")
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="Model path"
    )
    args = parser.parse_args(argv)

    try:
        artifact = load_model_artifact(args.model)
        records = pd.read_csv(args.input)
        predictions = predict_customers(records, artifact)
    except (FileNotFoundError, PredictionError, pd.errors.ParserError) as exc:
        print(f"Prediction failed: {exc}")
        return 2

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")

    print(predictions.to_string(index=False))
    print(
        f"\nThreshold: {artifact['threshold']:.2f} "
        f"| Model: {artifact.get('model_name', 'unknown')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Load and validate the UCI credit card default dataset."""

from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TARGET_COLUMN = "default"
ID_COLUMN = "ID"

SUPPORTED_SUFFIXES = {".xls", ".xlsx", ".csv"}
EXCEL_SUFFIXES = {".xls", ".xlsx"}
HEADER_PROBE_ROWS = 10
OFFICIAL_STEM = "default of credit card clients"

COLUMN_RENAMES = {
    "default payment next month": TARGET_COLUMN,
    "default.payment.next.month": TARGET_COLUMN,
    "PAY_0": "PAY_1",
}

PAY_COLUMNS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

REQUIRED_COLUMNS = [
    ID_COLUMN,
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_COLUMNS,
    *BILL_COLUMNS,
    *PAYMENT_COLUMNS,
    TARGET_COLUMN,
]

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)

MANUAL_DOWNLOAD_INSTRUCTIONS = f"""
Dataset not found. Download it manually:

1. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
2. Put the Excel file in: {RAW_DATA_DIR}
3. Re-run.
""".strip()


class DatasetNotFoundError(FileNotFoundError):
    pass


def _is_official_name(path: Path) -> bool:
    name = path.name.lower()
    return "credit" in name and "client" in name


def find_dataset_file(raw_dir: Path | str = RAW_DATA_DIR) -> Path:
    directory = Path(raw_dir)
    if not directory.exists():
        raise DatasetNotFoundError(
            f"Raw data dir missing: {directory}\n\n{MANUAL_DOWNLOAD_INSTRUCTIONS}"
        )

    candidates = [
        p
        for p in directory.iterdir()
        if p.is_file()
        and not p.name.startswith("~$")
        and not p.name.startswith(".")
        and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not candidates:
        raise DatasetNotFoundError(MANUAL_DOWNLOAD_INSTRUCTIONS)

    def preference(path: Path) -> tuple[int, str]:
        return (0 if path.suffix.lower() in EXCEL_SUFFIXES else 1, path.name.lower())

    exact = [p for p in candidates if p.stem.lower() == OFFICIAL_STEM]
    if exact:
        return min(exact, key=preference)

    official = [p for p in candidates if _is_official_name(p)]
    if len(official) == 1:
        return official[0]
    if len(official) > 1:
        names = sorted(p.name for p in official)
        raise DatasetNotFoundError(
            f"Multiple official-looking files in {directory}: {names}\n\n"
            f"{MANUAL_DOWNLOAD_INSTRUCTIONS}"
        )

    if len(candidates) == 1:
        return candidates[0]

    names = sorted(p.name for p in candidates)
    raise DatasetNotFoundError(
        f"Multiple candidates in {directory}: {names}\n\n{MANUAL_DOWNLOAD_INSTRUCTIONS}"
    )


def _find_header_row(probe: pd.DataFrame, source: Path | str = "<dataset>") -> int:
    for i in range(min(len(probe), HEADER_PROBE_ROWS)):
        values = {
            str(v).strip().upper()
            for v in probe.iloc[i].tolist()
            if not pd.isna(v)
        }
        if "ID" in values and "LIMIT_BAL" in values:
            return i
    raise ValueError(
        f"Could not find header with ID and LIMIT_BAL in first "
        f"{HEADER_PROBE_ROWS} rows of {source}"
    )


def _read_dataset_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported format '{suffix}' for {path}. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Empty file: {path}")

    reader = pd.read_csv if suffix == ".csv" else pd.read_excel
    try:
        probe = reader(path, header=None, nrows=HEADER_PROBE_ROWS)
    except Exception as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc

    header_row = _find_header_row(probe, path)
    try:
        frame = reader(path, header=header_row)
    except Exception as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc

    if frame.empty:
        raise ValueError(f"No data rows in {path}")
    if frame.shape[1] == 0:
        raise ValueError(f"No columns in {path}")
    return frame


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    stripped = [str(c).strip() for c in df.columns]
    if any(not n for n in stripped):
        raise ValueError("Empty column names after stripping")
    renamed = [COLUMN_RENAMES.get(n, n) for n in stripped]

    counts = {}
    for n in renamed:
        counts[n] = counts.get(n, 0) + 1
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    if duplicates:
        raise ValueError(f"Duplicate columns after rename: {duplicates}")

    result = df.copy()
    result.columns = renamed
    return result


def _validate_required_columns(df: pd.DataFrame, source: Path | str) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            f"{source} missing columns: {missing}. Got: {list(df.columns)}"
        )


def load_raw_dataset(
    dataset_path: Path | str | None = None,
    raw_dir: Path | str = RAW_DATA_DIR,
    download_if_missing: bool = False,
) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path else None
    if path is None:
        try:
            path = find_dataset_file(raw_dir)
        except DatasetNotFoundError:
            if not download_if_missing:
                raise
            path = download_dataset(raw_dir)

    if not path.exists():
        raise DatasetNotFoundError(
            f"File not found: {path}\n\n{MANUAL_DOWNLOAD_INSTRUCTIONS}"
        )

    df = standardize_columns(_read_dataset_file(path))
    _validate_required_columns(df, path)
    return df


def download_dataset(dest_dir: Path | str = RAW_DATA_DIR) -> Path:
    destination_dir = Path(dest_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "default of credit card clients.xls"

    if destination.exists():
        try:
            existing = standardize_columns(_read_dataset_file(destination))
            _validate_required_columns(existing, destination)
            return destination
        except (ValueError, OSError):
            pass

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (credit-card-default-prediction)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Download failed ({exc}).\n\n{MANUAL_DOWNLOAD_INSTRUCTIONS}"
        ) from exc

    if not payload:
        raise RuntimeError(
            f"Downloaded file was empty.\n\n{MANUAL_DOWNLOAD_INSTRUCTIONS}"
        )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=destination.suffix, dir=destination_dir
        ) as handle:
            handle.write(payload)
            temporary_path = Path(handle.name)

        try:
            downloaded = standardize_columns(_read_dataset_file(temporary_path))
            _validate_required_columns(downloaded, temporary_path)
        except (ValueError, OSError) as exc:
            raise RuntimeError(
                f"Downloaded file is invalid: {exc}\n\n{MANUAL_DOWNLOAD_INSTRUCTIONS}"
            ) from exc

        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

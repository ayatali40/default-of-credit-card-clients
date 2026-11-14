"""Tests for dataset discovery, header detection, and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader
from src.data_loader import (
    DatasetNotFoundError,
    download_dataset,
    find_dataset_file,
    load_raw_dataset,
    standardize_columns,
)


def test_find_dataset_file_prefers_official_name(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "other.xlsx").write_text("not a real workbook", encoding="utf-8")
    official = tmp_path / "default of credit card clients.xls"
    official.write_text("placeholder", encoding="utf-8")

    assert find_dataset_file(tmp_path) == official


def test_find_dataset_file_raises_when_empty(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError, match="download"):
        find_dataset_file(tmp_path)


def test_find_dataset_file_raises_for_missing_directory(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError):
        find_dataset_file(tmp_path / "does-not-exist")


def test_load_detects_header_row_and_renames_columns(
    tmp_path: Path, synthetic_raw_df: pd.DataFrame
):
    raw = synthetic_raw_df.copy().rename(
        columns={"PAY_1": "PAY_0", "default": "default payment next month"}
    )
    path = tmp_path / "default of credit card clients.xlsx"
    # Mimic the UCI layout: a metadata row above the real header.
    raw.to_excel(path, startrow=1, index=False)

    loaded = load_raw_dataset(dataset_path=path)

    assert len(loaded) == len(raw)
    assert "PAY_1" in loaded.columns and "PAY_0" not in loaded.columns
    assert "default" in loaded.columns
    assert "default payment next month" not in loaded.columns


def test_load_csv_dataset(tmp_path: Path, synthetic_raw_df: pd.DataFrame):
    path = tmp_path / "customers.csv"
    synthetic_raw_df.to_csv(path, index=False)

    loaded = load_raw_dataset(dataset_path=path)

    assert loaded.shape == synthetic_raw_df.shape
    assert loaded["default"].isin([0, 1]).all()


def test_load_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"ID": [1, 2], "LIMIT_BAL": [1000, 2000]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_dataset(dataset_path=path)


def test_load_missing_explicit_file_raises():
    with pytest.raises(DatasetNotFoundError):
        load_raw_dataset(dataset_path=Path("definitely/not/here.csv"))


def test_find_dataset_file_rejects_ambiguous_candidates(tmp_path: Path):
    (tmp_path / "first.csv").write_text("data", encoding="utf-8")
    (tmp_path / "second.csv").write_text("data", encoding="utf-8")

    with pytest.raises(DatasetNotFoundError, match="Multiple candidate"):
        find_dataset_file(tmp_path)


def test_find_dataset_file_rejects_multiple_official_lookalikes(tmp_path: Path):
    (tmp_path / "credit clients old.csv").write_text("data", encoding="utf-8")
    (tmp_path / "credit clients new.csv").write_text("data", encoding="utf-8")

    with pytest.raises(DatasetNotFoundError, match="Multiple official-looking"):
        find_dataset_file(tmp_path)


def test_load_rejects_missing_header_row(tmp_path: Path):
    path = tmp_path / "no_header.csv"
    pd.DataFrame([[1, 2, 3], [4, 5, 6]]).to_csv(path, header=False, index=False)

    with pytest.raises(ValueError, match="header row"):
        load_raw_dataset(dataset_path=path)


def test_load_rejects_empty_file(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_raw_dataset(dataset_path=path)


def test_load_rejects_unsupported_format(tmp_path: Path):
    path = tmp_path / "dataset.txt"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_raw_dataset(dataset_path=path)


def test_load_rejects_malformed_excel(tmp_path: Path):
    path = tmp_path / "broken.xlsx"
    path.write_text("this is not a workbook", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse"):
        load_raw_dataset(dataset_path=path)


def test_standardize_columns_detects_rename_collision():
    frame = pd.DataFrame({"ID": [1], "PAY_0": [0], "PAY_1": [0]})

    with pytest.raises(ValueError, match="Duplicate column names"):
        standardize_columns(frame)


def test_standardize_columns_detects_whitespace_duplicates():
    frame = pd.DataFrame({"ID": [1], "ID ": [2], "LIMIT_BAL": [3]})

    with pytest.raises(ValueError, match="Duplicate column names"):
        standardize_columns(frame)


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> _FakeResponse:  # noqa: PYI034
        return self

    def __exit__(self, *args) -> bool:
        return False


def test_download_reuses_an_existing_valid_dataset(
    tmp_path: Path, monkeypatch, synthetic_raw_df: pd.DataFrame
):
    destination = tmp_path / "default of credit card clients.xls"
    destination.write_bytes(b"existing-valid-placeholder")
    monkeypatch.setattr(
        data_loader, "_read_dataset_file", lambda path: synthetic_raw_df.copy()
    )

    def _fail(*args, **kwargs):
        raise AssertionError("urlopen must not be called for a valid dataset")

    monkeypatch.setattr(data_loader.urllib.request, "urlopen", _fail)

    result = download_dataset(tmp_path)

    assert result == destination
    assert destination.read_bytes() == b"existing-valid-placeholder"


def test_download_rejects_invalid_content_without_writing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        data_loader.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(b"not an excel workbook"),
    )

    with pytest.raises(RuntimeError, match="not a readable credit-card dataset"):
        download_dataset(tmp_path)

    assert not (tmp_path / "default of credit card clients.xls").exists()
    assert list(tmp_path.iterdir()) == []


def test_download_writes_validated_payload_atomically(
    tmp_path: Path, monkeypatch, synthetic_raw_df: pd.DataFrame
):
    monkeypatch.setattr(
        data_loader.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(b"validated-payload"),
    )
    monkeypatch.setattr(
        data_loader, "_read_dataset_file", lambda path: synthetic_raw_df.copy()
    )

    result = download_dataset(tmp_path)

    assert result.exists()
    assert result.read_bytes() == b"validated-payload"
    assert [path.name for path in tmp_path.iterdir()] == [result.name]

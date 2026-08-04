from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlpp.data import discover_csvs, read_csv_auto, read_table_auto
from mlpp.errors import DatasetError


@pytest.mark.parametrize("sep", [",", ";", "|", "\t"])
def test_read_csv_auto_detects_delimiter(tmp_path: Path, frame: pd.DataFrame, sep: str) -> None:
    path = tmp_path / "data.csv"
    frame.to_csv(path, sep=sep, index=False)
    loaded = read_csv_auto(path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)


def test_read_csv_auto_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="is empty"):
        read_csv_auto(path)


def test_read_csv_auto_rejects_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "headers.csv"
    path.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="zero rows"):
        read_csv_auto(path)


def test_read_csv_auto_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read"):
        read_csv_auto(tmp_path / "nope.csv")


@pytest.mark.parametrize("suffix", [".parquet", ".pq"])
def test_read_table_auto_round_trips_parquet(
    tmp_path: Path, frame: pd.DataFrame, suffix: str
) -> None:
    path = tmp_path / f"data{suffix}"
    frame.to_parquet(path, index=False)
    loaded = read_table_auto(path)
    assert list(loaded.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(loaded, frame)


@pytest.mark.parametrize("sep", [",", ";", "|", "\t"])
def test_read_table_auto_still_sniffs_csv_delimiters(
    tmp_path: Path, frame: pd.DataFrame, sep: str
) -> None:
    """Non-Parquet suffixes must reach `read_csv_auto` untouched."""
    path = tmp_path / "data.csv"
    frame.to_csv(path, sep=sep, index=False)
    loaded = read_table_auto(path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)


def test_read_table_auto_dispatches_on_suffix_case_insensitively(
    tmp_path: Path, frame: pd.DataFrame
) -> None:
    path = tmp_path / "data.PARQUET"
    frame.to_parquet(path, index=False)
    assert len(read_table_auto(path)) == len(frame)


def test_read_table_auto_rejects_corrupt_parquet(tmp_path: Path) -> None:
    """A pyarrow failure must surface as DatasetError, not as ArrowInvalid."""
    path = tmp_path / "corrupt.parquet"
    path.write_bytes(b"this is not a parquet file")
    with pytest.raises(DatasetError, match="cannot read"):
        read_table_auto(path)


def test_read_table_auto_rejects_empty_parquet(tmp_path: Path, frame: pd.DataFrame) -> None:
    path = tmp_path / "empty.parquet"
    frame.iloc[:0].to_parquet(path, index=False)
    with pytest.raises(DatasetError, match="zero rows"):
        read_table_auto(path)


def test_read_table_auto_reports_missing_parquet(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read"):
        read_table_auto(tmp_path / "absent.parquet")


def test_discover_csvs_is_sorted(tmp_path: Path) -> None:
    for name in ("b.csv", "a.csv", "c.csv", "notes.txt"):
        (tmp_path / name).write_text("x,y\n1,2\n", encoding="utf-8")
    assert [p.name for p in discover_csvs(tmp_path)] == ["a.csv", "b.csv", "c.csv"]


def test_discover_csvs_honours_explicit_order(tmp_path: Path) -> None:
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("x,y\n1,2\n", encoding="utf-8")
    assert [p.name for p in discover_csvs(tmp_path, ("b.csv", "a.csv"))] == ["b.csv", "a.csv"]


def test_discover_csvs_reports_missing_explicit_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="missing files"):
        discover_csvs(tmp_path, ("ghost.csv",))


def test_discover_csvs_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not a directory"):
        discover_csvs(tmp_path / "absent")


def test_discover_csvs_returns_empty_for_empty_directory(tmp_path: Path) -> None:
    assert discover_csvs(tmp_path) == []

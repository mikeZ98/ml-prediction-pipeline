from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlpp.data import discover_csvs, read_csv_auto
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

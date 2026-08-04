"""CSV/Parquet discovery and delimiter-tolerant loading. No TensorFlow import — keep it cheap."""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path

import pandas as pd

from mlpp.errors import DatasetError

_SNIFF_BYTES = 10_000
_DELIMITERS = ",;|\t"
_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read `path`, sniffing the delimiter and falling back to ';' then pandas' default.

    Raises DatasetError if the file cannot be parsed at all or has no rows.
    """
    for sep in _candidate_delimiters(path):
        try:
            df = pd.read_csv(path, sep=sep)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        # A single column usually means the delimiter guess was wrong.
        if df.shape[1] > 1:
            return _require_rows(df, path)
    msg = f"could not parse {path} with any of the candidate delimiters"
    raise DatasetError(msg)


def read_table_auto(path: Path) -> pd.DataFrame:
    """Read `path` as Parquet or CSV, dispatching on its suffix.

    An additive sibling of `read_csv_auto`, not a replacement: the training path
    still calls the CSV reader directly, so its delimiter sniffing is untouched by
    construction. Both branches raise DatasetError, so a caller handles one error
    type whatever the format.
    """
    if path.suffix.lower() in _PARQUET_SUFFIXES:
        return _read_parquet(path)
    return read_csv_auto(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file, translating pyarrow's failures into DatasetError.

    pyarrow raises `ArrowInvalid` — a ValueError subclass — for a corrupt file and
    OSError for an unreadable one. Catching the base classes keeps this module free
    of a direct pyarrow import while still promising callers a single error type.
    """
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    return _require_rows(df, path)


def _candidate_delimiters(path: Path) -> list[str | None]:
    """Sniffed delimiter first, then the historical fallbacks."""
    ordered: list[str | None] = []
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:_SNIFF_BYTES]
    except OSError as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    if not sample.strip():
        raise DatasetError(f"{path} is empty")
    with contextlib.suppress(csv.Error):
        ordered.append(csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter)
    for fallback in (";", ","):
        if fallback not in ordered:
            ordered.append(fallback)
    return ordered


def _require_rows(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df.empty:
        raise DatasetError(f"{path} parsed to zero rows")
    return df


def discover_csvs(directory: Path, explicit: tuple[str, ...] = ()) -> list[Path]:
    """List CSVs in `directory`, sorted. `explicit` selects specific filenames instead.

    Raises DatasetError when the directory is missing or an explicit file is absent.
    """
    if not directory.is_dir():
        raise DatasetError(f"not a directory: {directory}")
    if explicit:
        paths = [directory / name for name in explicit]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise DatasetError(f"missing files in {directory}: {missing}")
        return paths
    return sorted(directory.glob("*.csv"))

"""Inference payload capture: one JSON line per scored record, for drift analysis later.

The API hands every validated batch to a `PayloadSink` after scoring. The JSONL sink
writes one line per record:

    {"ts": "...+00:00", "request_id": "...", "session": "...",
     "features": {"feature_01": 0.5, ...}, "prediction": 101.3}

Files are named `payloads-<UTC date>-<host>-<pid>.jsonl`. Day-partitioning makes
"the last N days" a glob; host and pid in the name give every worker process its
own file, so several uvicorn workers or container replicas sharing one volume never
interleave partial lines.

Captured inputs are the *validated* records, so the store only ever holds rows the
model actually scored — never a malformed request.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol

import pandas as pd

from mlpp.errors import DatasetError

log = logging.getLogger(__name__)

PAYLOAD_GLOB: Final = "payloads-*.jsonl"
PREDICTION_COLUMN: Final = "prediction"
TIMESTAMP_COLUMN: Final = "ts"
REQUEST_ID_COLUMN: Final = "request_id"

FeatureRecord = Mapping[str, float | int | str]


class PayloadSink(Protocol):
    def write(
        self,
        records: Sequence[FeatureRecord],
        predictions: Sequence[float],
        *,
        request_id: str,
        session: str,
    ) -> None: ...


class JsonlPayloadSink:
    """Appends captured records to day-partitioned, per-process JSONL files.

    Thread-safe: the API calls `write` from worker threads. Each call opens, appends
    and closes, so a crash loses at most the batch being written, and an external
    rotation or cleanup never fights an open handle.
    """

    def __init__(self, directory: Path, *, clock: type[datetime] = datetime) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._origin = f"{_safe(socket.gethostname())}-{os.getpid()}"

    @property
    def directory(self) -> Path:
        return self._dir

    def write(
        self,
        records: Sequence[FeatureRecord],
        predictions: Sequence[float],
        *,
        request_id: str,
        session: str,
    ) -> None:
        if len(records) != len(predictions):
            raise ValueError(f"{len(records)} records but {len(predictions)} predictions")
        now = self._clock.now(UTC)
        stamp = now.isoformat(timespec="microseconds")
        lines = "".join(
            json.dumps(
                {
                    TIMESTAMP_COLUMN: stamp,
                    REQUEST_ID_COLUMN: request_id,
                    "session": session,
                    "features": dict(record),
                    PREDICTION_COLUMN: float(prediction),
                },
                allow_nan=False,
            )
            + "\n"
            for record, prediction in zip(records, predictions, strict=True)
        )
        path = self._dir / f"payloads-{now:%Y-%m-%d}-{self._origin}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(lines)


def read_payloads(source: Path) -> pd.DataFrame:
    """Load captured payloads from a `.jsonl` file or a directory of them.

    Returns one row per record: the feature columns, then `prediction`, `ts` and
    `request_id`. Unparseable lines — a write cut short by a crash — are skipped with
    a warning rather than failing the whole read.

    Raises DatasetError when nothing usable is found.
    """
    files = sorted(source.glob(PAYLOAD_GLOB)) if source.is_dir() else [source]
    if not files or not all(path.is_file() for path in files):
        raise DatasetError(f"no payload files at {source}")

    rows: list[dict[str, object]] = []
    skipped = 0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    row: dict[str, object] = dict(entry["features"])
                    row[PREDICTION_COLUMN] = float(entry[PREDICTION_COLUMN])
                    row[TIMESTAMP_COLUMN] = entry[TIMESTAMP_COLUMN]
                    row[REQUEST_ID_COLUMN] = entry[REQUEST_ID_COLUMN]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                rows.append(row)
    if skipped:
        log.warning("skipped %d unparseable payload line(s) under %s", skipped, source)
    if not rows:
        raise DatasetError(f"no usable payload records at {source}")
    return pd.DataFrame(rows)


def _safe(name: str) -> str:
    """A hostname reduced to filename-safe characters."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "host"

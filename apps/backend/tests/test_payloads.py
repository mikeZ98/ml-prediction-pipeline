"""Payload capture. Stdlib + pandas only — part of the fast suite."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mlpp.errors import DatasetError
from mlpp.monitoring.payloads import JsonlPayloadSink, read_payloads

RECORDS = [
    {"feature_01": 0.5, "comp_active": 1, "site": "north"},
    {"feature_01": -1.0, "comp_active": 0, "site": "south"},
]


class _FrozenClock(datetime):
    moment = datetime(2026, 10, 22, 23, 59, 59, tzinfo=UTC)

    @classmethod
    def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
        return cls.moment


def test_a_batch_round_trips_with_its_metadata(tmp_path: Path) -> None:
    sink = JsonlPayloadSink(tmp_path)
    sink.write(RECORDS, [101.5, 98.25], request_id="req-1", session="example")

    frame = read_payloads(tmp_path)
    assert frame[["feature_01", "comp_active", "site"]].to_dict("records") == RECORDS
    assert frame["prediction"].tolist() == [101.5, 98.25]
    assert frame["request_id"].tolist() == ["req-1", "req-1"]


def test_files_are_partitioned_by_utc_day_and_process(tmp_path: Path) -> None:
    sink = JsonlPayloadSink(tmp_path, clock=_FrozenClock)
    sink.write(RECORDS[:1], [1.0], request_id="r", session="s")
    (path,) = tmp_path.glob("*.jsonl")
    assert path.name.startswith("payloads-2026-10-22-")
    assert path.name.endswith(f"-{os.getpid()}.jsonl")
    assert json.loads(path.read_text(encoding="utf-8"))["ts"].startswith("2026-10-22T23:59:59")


def test_writes_append_and_reads_span_every_file(tmp_path: Path) -> None:
    JsonlPayloadSink(tmp_path).write(RECORDS, [1.0, 2.0], request_id="a", session="s")
    JsonlPayloadSink(tmp_path).write(RECORDS, [3.0, 4.0], request_id="b", session="s")
    other_day = tmp_path / "payloads-2026-01-01-otherhost-1.jsonl"
    other_day.write_text(
        json.dumps(
            {"ts": "t", "request_id": "c", "session": "s", "features": RECORDS[0], "prediction": 5}
        )
        + "\n",
        encoding="utf-8",
    )
    assert sorted(read_payloads(tmp_path)["prediction"]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_single_file_can_be_read_directly(tmp_path: Path) -> None:
    JsonlPayloadSink(tmp_path).write(RECORDS, [1.0, 2.0], request_id="a", session="s")
    (path,) = tmp_path.glob("*.jsonl")
    assert len(read_payloads(path)) == 2


def test_a_truncated_line_is_skipped_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sink = JsonlPayloadSink(tmp_path)
    sink.write(RECORDS, [1.0, 2.0], request_id="a", session="s")
    (path,) = tmp_path.glob("*.jsonl")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"ts": "cut short by a cra\n{"no": "features"}\n\n')

    assert len(read_payloads(tmp_path)) == 2
    assert "skipped 2 unparseable" in caplog.text


@pytest.mark.parametrize("content", ["", "not json\n", '{"features": {}}\n'])
def test_no_usable_records_is_a_dataset_error(tmp_path: Path, content: str) -> None:
    (tmp_path / "payloads-2026-10-22-h-1.jsonl").write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match="no usable payload records"):
        read_payloads(tmp_path)


def test_an_empty_or_missing_location_is_a_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no payload files"):
        read_payloads(tmp_path)
    with pytest.raises(DatasetError, match="no payload files"):
        read_payloads(tmp_path / "absent.jsonl")


def test_records_and_predictions_must_pair_up(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2 records but 1 predictions"):
        JsonlPayloadSink(tmp_path).write(RECORDS, [1.0], request_id="a", session="s")


def test_non_finite_predictions_are_refused_rather_than_written_as_invalid_json(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        JsonlPayloadSink(tmp_path).write(RECORDS[:1], [float("nan")], request_id="a", session="s")


def test_concurrent_writers_never_interleave_lines(tmp_path: Path) -> None:
    sink = JsonlPayloadSink(tmp_path)
    batch = RECORDS * 50

    def worker(index: int) -> None:
        for _ in range(20):
            sink.write(batch, [float(index)] * len(batch), request_id=str(index), session="s")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    frame = read_payloads(tmp_path)
    assert len(frame) == 8 * 20 * len(batch)
    assert frame.groupby("request_id")["prediction"].nunique().eq(1).all()

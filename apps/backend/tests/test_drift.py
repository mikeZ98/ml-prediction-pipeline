"""Drift detection. Report parsing and input checks are fast; real Evidently runs are slow."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlpp.errors import DatasetError, SchemaError
from mlpp.monitoring.drift import detect_drift, summarise_snapshot
from mlpp.monitoring.drift_cli import EXIT_DRIFT, EXIT_ERROR, EXIT_OK, main
from mlpp.monitoring.payloads import JsonlPayloadSink
from mlpp.session import FeatureContract, SessionWriter

NUMERIC = ("feature_01", "feature_02")
CATEGORICAL = ("site",)
CONTRACT = FeatureContract(NUMERIC, CATEGORICAL, "target", (*NUMERIC, "site_a", "site_b"))


def _frame(
    rng: np.random.Generator, n: int, *, shift: float = 0.0, p_a: float = 0.5
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_01": rng.normal(shift, 1.0, size=n),
            "feature_02": rng.normal(0.0, 1.0, size=n),
            "site": rng.choice(["a", "b"], size=n, p=[p_a, 1 - p_a]),
            "target": rng.normal(size=n),
        }
    )


# --- fast: parsing Evidently's report ---------------------------------------------------


class _Status(Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


def _snapshot(*columns: tuple[str, float, bool], share: float | None = None) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for index, (column, score, drifted) in enumerate(columns):
        params = {
            "type": "evidently:metric_v2:ValueDrift",
            "column": column,
            "method": "K-S p_value",
            "threshold": 0.05,
        }
        metrics.append({"id": f"m{index}", "config": params, "value": score})
        tests.append(
            {
                "id": "drift",
                "metric_config": {"metric_id": f"m{index}", "params": params},
                "status": _Status.FAIL if drifted else _Status.SUCCESS,
            }
        )
    if share is not None:
        metrics.append(
            {
                "id": "count",
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                "value": {"count": round(share * len(columns)), "share": share},
            }
        )
    return {"metrics": metrics, "tests": tests}


def test_column_verdicts_come_from_the_test_status_not_the_score() -> None:
    """A p-value drifts below its threshold, a distance above it — so trust the test."""
    result = summarise_snapshot(
        _snapshot(("feature_01", 1e-9, True), ("feature_02", 0.8, False), ("site", 0.01, False)),
        drift_share=0.5,
        n_reference=100,
        n_current=50,
    )
    assert [(c.column, c.drifted) for c in result.columns] == [
        ("feature_01", True),
        ("feature_02", False),
        ("site", False),
    ]
    assert result.drifted_columns == 1
    assert result.drifted_share == pytest.approx(1 / 3)
    assert result.dataset_drift is False


def test_dataset_drift_is_share_at_or_above_the_threshold() -> None:
    snapshot = _snapshot(("a", 0.0, True), ("b", 0.9, False), share=0.5)
    assert summarise_snapshot(snapshot, drift_share=0.5, n_reference=1, n_current=1).dataset_drift


@pytest.mark.parametrize("status", ["FAIL", "TestStatus.FAIL", _Status.FAIL])
def test_status_is_accepted_as_enum_value_or_name(status: object) -> None:
    snapshot = _snapshot(("a", 0.0, True))
    snapshot["tests"][0]["status"] = status
    result = summarise_snapshot(snapshot, drift_share=0.5, n_reference=1, n_current=1)
    assert result.columns[0].drifted


def test_a_report_without_column_results_is_rejected() -> None:
    with pytest.raises(ValueError, match="no per-column drift"):
        summarise_snapshot(
            {"metrics": [], "tests": []}, drift_share=0.5, n_reference=1, n_current=1
        )


def test_missing_input_columns_are_a_schema_error(rng: np.random.Generator) -> None:
    current = _frame(rng, 100).drop(columns=["site"])
    with pytest.raises(SchemaError, match="current data is missing input columns: \\['site'\\]"):
        detect_drift(_frame(rng, 100), current, CONTRACT)


def test_too_few_rows_is_a_dataset_error(rng: np.random.Generator) -> None:
    with pytest.raises(DatasetError, match="current data has 10 row"):
        detect_drift(_frame(rng, 100), _frame(rng, 10), CONTRACT, min_rows=30)


def test_drift_share_must_be_a_fraction(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="drift_share"):
        detect_drift(_frame(rng, 100), _frame(rng, 100), CONTRACT, drift_share=0.0)


# --- slow: real Evidently ---------------------------------------------------------------


@pytest.mark.slow
def test_same_distribution_shows_no_drift(rng: np.random.Generator) -> None:
    result = detect_drift(_frame(rng, 2000), _frame(rng, 500), CONTRACT)
    assert not result.dataset_drift
    assert result.drifted == ()
    assert {c.column for c in result.columns} == set(NUMERIC + CATEGORICAL)
    assert (result.n_reference, result.n_current) == (2000, 500)


@pytest.mark.slow
def test_a_shifted_numeric_and_a_skewed_categorical_are_caught(
    rng: np.random.Generator, tmp_path: Path
) -> None:
    html = tmp_path / "report" / "drift.html"
    result = detect_drift(
        _frame(rng, 2000), _frame(rng, 500, shift=3.0, p_a=0.95), CONTRACT, html_path=html
    )
    assert {c.column for c in result.drifted} == {"feature_01", "site"}
    assert result.dataset_drift
    assert html.is_file() and html.stat().st_size > 0


@pytest.mark.slow
def test_categorical_levels_compare_across_dtypes(rng: np.random.Generator) -> None:
    """A CSV's int level and a payload's string level are the same category."""
    contract = FeatureContract(NUMERIC, ("code",), "target", (*NUMERIC, "code_1", "code_2"))
    reference = _frame(rng, 1000).assign(code=rng.choice([1, 2], size=1000))
    current = _frame(rng, 300).assign(code=rng.choice(["1", "2"], size=300))
    result = detect_drift(reference, current, contract)
    assert not next(c for c in result.columns if c.column == "code").drifted


# --- slow: the CLI ----------------------------------------------------------------------


def _session(tmp_path: Path) -> Path:
    writer = SessionWriter(tmp_path / "session")
    writer.set_features(CONTRACT)
    writer.flush()
    return writer.session_dir


def _payloads(directory: Path, frame: pd.DataFrame) -> Path:
    sink = JsonlPayloadSink(directory)
    records = frame[list(NUMERIC + CATEGORICAL)].to_dict("records")
    sink.write(records, [0.0] * len(records), request_id="r", session="session")
    return directory


@pytest.mark.slow
# The current sample is as large as the reference on purpose: at a few hundred rows,
# Evidently's default normed-Wasserstein threshold (0.1) sits near the 95th percentile
# of the no-drift statistic, so a single column false-positives ~10% of the time.
@pytest.mark.parametrize(
    ("shift", "p_a", "expected"), [(0.0, 0.5, EXIT_OK), (3.0, 0.95, EXIT_DRIFT)]
)
def test_cli_exit_status_is_the_verdict(
    tmp_path: Path,
    rng: np.random.Generator,
    shift: float,
    p_a: float,
    expected: int,
    capsys: Any,
) -> None:
    reference_dir = tmp_path / "TRAIN"
    reference_dir.mkdir()
    _frame(rng, 1000).to_csv(reference_dir / "train_01.csv", index=False)
    _frame(rng, 1000).to_csv(reference_dir / "train_02.csv", index=False)
    payload_dir = _payloads(tmp_path / "PAYLOADS", _frame(rng, 2000, shift=shift, p_a=p_a))
    summary = tmp_path / "drift.json"

    code = main(
        [
            "--session",
            str(_session(tmp_path)),
            "--reference",
            str(reference_dir),
            "--current",
            str(payload_dir),
            "--json",
            str(summary),
        ]
    )

    assert code == expected, capsys.readouterr().out
    written = json.loads(summary.read_text(encoding="utf-8"))
    assert written["n_reference"] == 2000
    assert written["n_current"] == 2000
    assert written["dataset_drift"] is (expected == EXIT_DRIFT)


def test_cli_reports_unusable_input_as_an_error(tmp_path: Path, capsys: Any) -> None:
    code = main(["--session", str(_session(tmp_path)), "--current", str(tmp_path / "PAYLOADS")])
    assert code == EXIT_ERROR
    assert "does not exist" in capsys.readouterr().err

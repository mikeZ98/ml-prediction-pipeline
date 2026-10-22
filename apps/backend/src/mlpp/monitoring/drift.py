"""Data drift: do live inference inputs still look like the data the model learned from?

Compares the captured payloads (`mlpp.monitoring.payloads`) against the training
dataset, column by column, with Evidently's `DataDriftPreset`. Evidently picks a
statistical test per column from its type and the sample size (Kolmogorov–Smirnov
or normed Wasserstein distance for numerics, chi-squared / Z-test or Jensen–Shannon
for categoricals); a column *drifts* when its test fails, and the dataset drifts
when at least `drift_share` of the columns do.

Only the model's *input* columns are compared, taken from the session's feature
contract — the manifest is the single source of truth for which columns those are.
The target is absent from live traffic and predictions are absent from the training
data, so neither has a like-for-like reference.

`summarise_snapshot` turns Evidently's report dictionary into a typed `DriftResult`
and is pure, so its parsing is tested without importing Evidently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from mlpp.errors import DatasetError, SchemaError
from mlpp.session import FeatureContract

#: Below this, a two-sample test says more about noise than about drift.
DEFAULT_MIN_ROWS: Final = 30
DEFAULT_DRIFT_SHARE: Final = 0.5

_VALUE_DRIFT: Final = "evidently:metric_v2:ValueDrift"
_DRIFTED_COUNT: Final = "evidently:metric_v2:DriftedColumnsCount"


@dataclass(frozen=True, slots=True)
class ColumnDrift:
    column: str
    #: The statistical test Evidently chose, e.g. "K-S p_value".
    method: str
    #: The test's statistic: a p-value for tests, a distance for distance methods.
    score: float
    threshold: float
    drifted: bool


@dataclass(frozen=True, slots=True)
class DriftResult:
    dataset_drift: bool
    drifted_columns: int
    drifted_share: float
    drift_share_threshold: float
    columns: tuple[ColumnDrift, ...]
    n_reference: int
    n_current: int

    @property
    def drifted(self) -> tuple[ColumnDrift, ...]:
        return tuple(c for c in self.columns if c.drifted)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    contract: FeatureContract,
    *,
    drift_share: float = DEFAULT_DRIFT_SHARE,
    min_rows: int = DEFAULT_MIN_ROWS,
    html_path: Path | None = None,
) -> DriftResult:
    """Run Evidently's drift preset over the contract's input columns.

    Raises SchemaError when either frame lacks an input column, DatasetError when
    either has fewer than `min_rows` rows, and ValueError for a `drift_share`
    outside (0, 1]. Writes Evidently's interactive report to `html_path` if given.
    """
    if not 0.0 < drift_share <= 1.0:
        raise ValueError(f"drift_share must be in (0, 1], got {drift_share}")
    columns = list(contract.numeric_columns + contract.categorical_columns)
    ref = _select(reference, columns, contract, "reference", min_rows)
    cur = _select(current, columns, contract, "current", min_rows)

    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(
        numerical_columns=list(contract.numeric_columns),
        categorical_columns=list(contract.categorical_columns),
    )
    report = Report([DataDriftPreset(columns=columns, drift_share=drift_share)], include_tests=True)
    snapshot = report.run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition),
    )
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    return summarise_snapshot(
        snapshot.dict(), drift_share=drift_share, n_reference=len(ref), n_current=len(cur)
    )


def summarise_snapshot(
    payload: Mapping[str, Any], *, drift_share: float, n_reference: int, n_current: int
) -> DriftResult:
    """Project an Evidently `Snapshot.dict()` onto `DriftResult`.

    Per-column verdicts come from the preset's tests (status FAIL = drifted) rather
    than from re-deriving them from the score, because the direction of the
    comparison depends on the method: a p-value drifts *below* its threshold, a
    distance *above* it.
    """
    metrics = {m["id"]: m for m in payload.get("metrics", [])}
    columns: list[ColumnDrift] = []
    for test in payload.get("tests", []):
        config = test.get("metric_config", {})
        params = config.get("params", {})
        if params.get("type") != _VALUE_DRIFT:
            continue
        metric = metrics.get(config.get("metric_id"))
        if metric is None:
            continue
        columns.append(
            ColumnDrift(
                column=str(params["column"]),
                method=str(params.get("method", "")),
                score=float(metric["value"]),
                threshold=float(params["threshold"]),
                drifted=_failed(test.get("status")),
            )
        )
    if not columns:
        raise ValueError("the report contains no per-column drift results")

    count = sum(c.drifted for c in columns)
    share = count / len(columns)
    for metric in metrics.values():  # prefer Evidently's own aggregate when present
        if metric.get("config", {}).get("type") == _DRIFTED_COUNT:
            count = int(metric["value"]["count"])
            share = float(metric["value"]["share"])
    return DriftResult(
        dataset_drift=share >= drift_share,
        drifted_columns=count,
        drifted_share=share,
        drift_share_threshold=drift_share,
        columns=tuple(sorted(columns, key=lambda c: c.column)),
        n_reference=n_reference,
        n_current=n_current,
    )


def _select(
    frame: pd.DataFrame,
    columns: Sequence[str],
    contract: FeatureContract,
    label: str,
    min_rows: int,
) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaError(f"{label} data is missing input columns: {missing}")
    if len(frame) < min_rows:
        raise DatasetError(
            f"{label} data has {len(frame)} row(s); drift needs at least {min_rows} "
            "for the statistical tests to mean anything"
        )
    selected = frame[list(columns)].copy()
    for column in contract.numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    # One dtype per categorical on both sides: a CSV's int 1 and a payload's "1" are
    # the same level.
    for column in contract.categorical_columns:
        selected[column] = selected[column].astype("string")
    return selected


def _failed(status: object) -> bool:
    """Evidently returns a `TestStatus` enum; accept its value or name as a string too."""
    raw = getattr(status, "value", status)
    return str(raw).upper().rsplit(".", 1)[-1] == "FAIL"

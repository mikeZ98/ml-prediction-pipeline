"""Regression metrics. Pure NumPy/sklearn — no TensorFlow, no plotting, no I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """MSE/RMSE/MAE/R² over flattened inputs.

    Raises ValueError on length mismatch or empty input, rather than letting
    sklearn surface it further down the call stack.
    """
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")

    mse = float(mean_squared_error(true, pred))
    return RegressionMetrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(true, pred)),
        r2=float(r2_score(true, pred)),
    )


def rolling_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rolling mean of (pred - true).

    With `active_mask`, inactive samples are held at the last active value before
    averaging, matching the notebook's "rolling over active samples only" view.
    """

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    diffs = (
        np.asarray(y_pred, dtype=np.float64).ravel() - np.asarray(y_true, dtype=np.float64).ravel()
    )
    series = pd.Series(diffs)
    if active_mask is not None:
        mask = np.asarray(active_mask).ravel().astype(bool)
        if mask.shape != diffs.shape:
            raise ValueError(f"active_mask shape {mask.shape} != data shape {diffs.shape}")
        series = pd.Series(np.where(mask, diffs, np.nan)).ffill()
    return np.asarray(series.rolling(window=window, min_periods=1).mean().to_numpy())

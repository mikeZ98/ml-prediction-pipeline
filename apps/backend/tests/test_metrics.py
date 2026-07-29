from __future__ import annotations

import numpy as np
import pytest

from mlpp.metrics import regression_metrics, rolling_error


def test_perfect_prediction_scores_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m.mse == 0.0
    assert m.rmse == 0.0
    assert m.mae == 0.0
    assert m.r2 == 1.0


def test_metrics_match_hand_computed_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])  # errors: +1, 0, +2
    m = regression_metrics(y_true, y_pred)
    assert m.mse == pytest.approx(5 / 3)
    assert m.rmse == pytest.approx(np.sqrt(5 / 3))
    assert m.mae == pytest.approx(1.0)


def test_metrics_flatten_column_vectors() -> None:
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert regression_metrics(y_true, y_pred).r2 == 1.0


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="empty arrays"):
        regression_metrics(np.array([]), np.array([]))


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_as_dict_exposes_all_four_metrics() -> None:
    keys = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0])).as_dict().keys()
    assert set(keys) == {"mse", "rmse", "mae", "r2"}


def test_rolling_error_averages_residuals() -> None:
    y_true = np.zeros(4)
    y_pred = np.array([1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(rolling_error(y_true, y_pred, window=2), [1.0, 2.0, 4.0, 6.0])


def test_rolling_error_is_full_length() -> None:
    """min_periods=1 means no leading NaNs, unlike the notebook's raw rolling()."""
    out = rolling_error(np.zeros(10), np.ones(10), window=500)
    assert out.shape == (10,)
    assert np.isfinite(out).all()


def test_rolling_error_holds_inactive_samples() -> None:
    y_true = np.zeros(4)
    y_pred = np.array([2.0, 100.0, 4.0, 100.0])
    mask = np.array([True, False, True, False])
    out = rolling_error(y_true, y_pred, window=1, active_mask=mask)
    np.testing.assert_allclose(out, [2.0, 2.0, 4.0, 4.0])


def test_rolling_error_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        rolling_error(np.zeros(3), np.zeros(3), window=0)


def test_rolling_error_rejects_mask_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="active_mask shape"):
        rolling_error(np.zeros(4), np.zeros(4), window=2, active_mask=np.array([True, False]))

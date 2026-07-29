from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mlpp.config import DataConfig
from mlpp.errors import NotFittedError, SchemaError
from mlpp.preprocess import Preprocessor, align_columns, resolve_schema


def test_align_columns_orders_by_config(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    shuffled = frame[["target", "comp_active", "feature_02", "feature_01"]]
    aligned = align_columns(shuffled, data_cfg)
    assert list(aligned.columns) == ["feature_01", "feature_02", "comp_active", "target"]


def test_align_columns_does_not_mutate_caller(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    lenient = replace(data_cfg, strict_schema=False)
    original = frame.drop(columns=["feature_02"])
    snapshot = original.copy()
    align_columns(original, lenient)
    pd.testing.assert_frame_equal(original, snapshot)


def test_strict_schema_rejects_missing_column(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    with pytest.raises(SchemaError, match="feature_02"):
        align_columns(frame.drop(columns=["feature_02"]), data_cfg)


def test_lenient_schema_fills_missing_column(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    lenient = replace(data_cfg, strict_schema=False)
    aligned = align_columns(frame.drop(columns=["feature_02"]), lenient)
    assert (aligned["feature_02"] == 0.0).all()


def test_resolve_schema_splits_on_dtype(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    frame["comp_active"] = frame["comp_active"].astype(str)
    schema = resolve_schema(frame, data_cfg)
    assert schema.numeric == ("feature_01", "feature_02")
    assert schema.categorical == ("comp_active",)


def test_resolve_schema_honours_forced_categoricals(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    cfg = replace(data_cfg, categorical_columns=("comp_active",))
    schema = resolve_schema(frame, cfg)
    assert schema.categorical == ("comp_active",), "numeric dtype must not override the config"


def test_transform_standardises_features(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    x, y = Preprocessor(data_cfg).fit_transform(frame)
    assert x.shape == (len(frame), 3, 1)
    assert y is not None and y.shape == (len(frame), 1)
    np.testing.assert_allclose(x[:, :, 0].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(x[:, :, 0].std(axis=0), 1.0, atol=1e-5)


def test_test_data_reuses_train_statistics(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    """A shifted test set must NOT re-centre on its own mean — that would leak."""
    pre = Preprocessor(data_cfg).fit(frame)
    shifted = frame.copy()
    shifted["feature_01"] += 10.0
    x_test, _ = pre.transform(shifted)
    assert x_test[:, 0, 0].mean() > 5.0


def test_feature_order_is_stable_across_dtype_drift(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    """The notebook re-derived the split per file; a dtype change reordered features."""
    cfg = replace(data_cfg, categorical_columns=("comp_active",))
    pre = Preprocessor(cfg).fit(frame)
    drifted = frame.copy()
    drifted["comp_active"] = drifted["comp_active"].astype(str)
    x_train, _ = pre.transform(frame)
    x_test, _ = pre.transform(drifted)
    assert x_train.shape[1] == x_test.shape[1] == pre.n_features


def test_unknown_category_is_encoded_as_zeros(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    cfg = replace(data_cfg, categorical_columns=("comp_active",))
    pre = Preprocessor(cfg).fit(frame)
    unseen = frame.copy()
    unseen["comp_active"] = 99
    x, _ = pre.transform(unseen)
    onehot_block = x[:, len(pre.schema.numeric) :, 0]
    assert onehot_block.sum() == 0.0


def test_transform_returns_none_target_when_absent(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    pre = Preprocessor(data_cfg).fit(frame)
    x, y = pre.transform(frame.drop(columns=["target"]))
    assert y is None
    assert x.shape[0] == len(frame)


def test_inverse_target_round_trips(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    pre = Preprocessor(data_cfg).fit(frame)
    _, y = pre.transform(frame)
    np.testing.assert_allclose(pre.inverse_target(y), frame["target"].to_numpy(), atol=1e-6)


def test_numeric_nan_filled_with_mean(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    holed = frame.copy()
    holed.loc[0, "feature_01"] = np.nan
    x, _ = Preprocessor(data_cfg).fit_transform(holed)
    assert np.isfinite(x).all()


def test_index_of_uses_expanded_feature_names(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    """comp_active sits at config index 2, but after one-hot the layout shifts."""
    cfg = replace(data_cfg, categorical_columns=("feature_02",))
    pre = Preprocessor(cfg).fit(frame.assign(feature_02="a"))
    assert pre.index_of("comp_active") == pre.feature_names.index("comp_active")
    with pytest.raises(KeyError):
        pre.index_of("feature_02")


def test_transform_before_fit_raises(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    with pytest.raises(NotFittedError):
        Preprocessor(data_cfg).transform(frame)


def test_fit_requires_target_column(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    with pytest.raises(SchemaError, match="required to fit"):
        Preprocessor(data_cfg).fit(frame.drop(columns=["target"]))


def test_fitted_state_exposes_the_estimators(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    state = Preprocessor(data_cfg).fit(frame).fitted_state
    assert state.scaler is not None
    assert state.output_scaler is not None
    assert state.onehot is not None


def test_fitted_state_carries_no_schema(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    """The schema belongs to the manifest; duplicating it here is what caused drift."""
    state = Preprocessor(data_cfg).fit(frame).fitted_state
    assert not hasattr(state, "schema")
    assert not hasattr(state, "feature_names")


def test_fitted_state_before_fit_raises(data_cfg: DataConfig) -> None:
    with pytest.raises(NotFittedError):
        _ = Preprocessor(data_cfg).fitted_state


def test_restore_reproduces_transforms(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    pre = Preprocessor(data_cfg).fit(frame)
    restored = Preprocessor.restore(data_cfg, pre.fitted_state, pre.schema, pre.feature_names)
    assert restored.feature_names == pre.feature_names
    np.testing.assert_allclose(restored.transform(frame)[0], pre.transform(frame)[0])

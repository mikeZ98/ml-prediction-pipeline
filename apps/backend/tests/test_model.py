"""Keras-dependent tests. Slow because importing TensorFlow costs seconds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlpp.config import ColumnConfig, TrainConfig
from mlpp.preprocess import Preprocessor

pytestmark = pytest.mark.slow


def test_model_output_shape_is_scalar_per_row() -> None:
    from mlpp.model import build_model

    model = build_model(5, TrainConfig())
    assert model.input_shape == (None, 5, 1)
    assert model.output_shape == (None, 1)


def test_model_rejects_zero_features() -> None:
    from mlpp.model import build_model

    with pytest.raises(ValueError, match="n_features must be >= 1"):
        build_model(0, TrainConfig())


@pytest.mark.parametrize("loss", ["mse", "huber"])
def test_model_compiles_for_each_loss(loss: str) -> None:
    from mlpp.model import build_model

    model = build_model(3, TrainConfig(loss=loss))  # type: ignore[arg-type]
    assert model.loss is not None


def test_set_seed_makes_training_reproducible(
    frame: pd.DataFrame, column_cfg: ColumnConfig
) -> None:
    from mlpp.model import build_model
    from mlpp.training import set_seed

    x, y = Preprocessor(column_cfg).fit_transform(frame)
    weights = []
    for _ in range(2):
        set_seed(7)
        model = build_model(x.shape[1], TrainConfig())
        model.fit(x, y, epochs=1, batch_size=16, verbose=0)
        weights.append(model.get_weights()[0])
    np.testing.assert_allclose(weights[0], weights[1])


def test_active_sample_weights_are_disabled_by_default(
    frame: pd.DataFrame, column_cfg: ColumnConfig
) -> None:
    from mlpp.training import active_sample_weights

    pre = Preprocessor(column_cfg).fit(frame)
    x, _ = pre.transform(frame)
    assert active_sample_weights(x, pre, TrainConfig()) is None


def test_active_sample_weights_track_the_right_column(
    frame: pd.DataFrame, column_cfg: ColumnConfig
) -> None:
    from mlpp.training import active_sample_weights

    pre = Preprocessor(column_cfg).fit(frame)
    x, _ = pre.transform(frame)
    weights = active_sample_weights(x, pre, TrainConfig(active_weight_alpha=0.5))

    assert weights is not None
    scaled_active = x[:, pre.index_of("comp_active"), 0]
    np.testing.assert_allclose(weights, 1.0 + 0.5 * (scaled_active != 0))


def test_active_sample_weights_none_when_column_absent(
    frame: pd.DataFrame, column_cfg: ColumnConfig
) -> None:
    from mlpp.training import active_sample_weights

    pre = Preprocessor(column_cfg).fit(frame)
    x, _ = pre.transform(frame)
    cfg = TrainConfig(active_weight_alpha=0.5, active_column="does_not_exist")
    assert active_sample_weights(x, pre, cfg) is None


def test_train_rejects_mismatched_lengths(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path
) -> None:
    from mlpp.model import build_model
    from mlpp.training import train

    x, y = Preprocessor(column_cfg).fit_transform(frame)
    model = build_model(x.shape[1], TrainConfig())
    with pytest.raises(ValueError, match="rows but y has"):
        train(model, x, y[:-1], tmp_path, TrainConfig(epochs=1), verbose=0)


def test_train_writes_checkpoint_and_log(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path
) -> None:
    from mlpp.model import build_model
    from mlpp.training import BEST_MODEL_FILE, TRAINING_LOG_FILE, train

    x, y = Preprocessor(column_cfg).fit_transform(frame)
    cfg = TrainConfig(epochs=1, batch_size=16, validation_split=0.2)
    model = build_model(x.shape[1], cfg)
    history = train(model, x, y, tmp_path, cfg, verbose=0)

    assert "loss" in history.history
    assert (tmp_path / BEST_MODEL_FILE).is_file()
    assert (tmp_path / TRAINING_LOG_FILE).is_file()

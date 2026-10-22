"""PyTorch model, fit loop and ONNX export. Slow because importing torch costs seconds."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlpp.config import ColumnConfig, TrainConfig
from mlpp.errors import ArtifactError, TrainingError
from mlpp.preprocess import Preprocessor

pytestmark = pytest.mark.slow

#: The Keras network's parameter count. The port is layer-for-layer, so it must match.
KERAS_PARAMETER_COUNT_13_FEATURES = 289_473


def _xy(frame: pd.DataFrame, column_cfg: ColumnConfig) -> tuple[np.ndarray, np.ndarray]:
    x, y = Preprocessor(column_cfg).fit_transform(frame)
    assert y is not None
    return x, y


def _fit(model, x, y, tmp_path: Path, **overrides):  # type: ignore[no-untyped-def]
    from mlpp.training.loop import fit

    cfg = TrainConfig(**{"epochs": 2, "batch_size": 16, "validation_split": 0.2, **overrides})
    return fit(
        model,
        x,
        y,
        cfg,
        checkpoint_path=tmp_path / "best.pt",
        log_path=tmp_path / "log.csv",
        verbose=0,
    )


# --- model ------------------------------------------------------------------------------


def test_model_output_is_one_scalar_per_row() -> None:
    import torch

    from mlpp.training.model import build_model

    out = build_model(5).eval()(torch.zeros(4, 5, 1))
    assert tuple(out.shape) == (4, 1)


def test_model_rejects_zero_features() -> None:
    from mlpp.training.model import build_model

    with pytest.raises(ValueError, match="n_features must be >= 1"):
        build_model(0)


def test_port_has_exactly_the_keras_parameter_count() -> None:
    """GRU gate layouts differ between frameworks; the parameter count must not."""
    from mlpp.training.model import build_model, count_parameters

    assert count_parameters(build_model(13)) == KERAS_PARAMETER_COUNT_13_FEATURES


def test_l2_penalty_covers_the_gru_input_kernels_only() -> None:
    import torch

    from mlpp.training.model import GRU_KERNEL_L2, build_model

    model = build_model(3)
    expected = sum(
        float(p.detach().pow(2).sum())
        for gru in (model.gru1, model.gru2)
        for name, p in gru.named_parameters()
        if name.startswith("weight_ih")
    )
    assert float(model.l2_penalty().detach()) == pytest.approx(GRU_KERNEL_L2 * expected, rel=1e-5)

    with torch.no_grad():
        model.gru1.weight_hh_l0.mul_(10.0)  # the recurrent kernel is not regularised
    assert float(model.l2_penalty().detach()) == pytest.approx(GRU_KERNEL_L2 * expected, rel=1e-5)


@pytest.mark.parametrize("loss", ["mse", "huber"])
def test_each_loss_is_per_element(loss: str) -> None:
    import torch

    from mlpp.training.model import make_loss

    out = make_loss(loss)(torch.zeros(4, 1), torch.ones(4, 1))  # type: ignore[arg-type]
    assert tuple(out.shape) == (4, 1)


# --- fit loop ---------------------------------------------------------------------------


def test_set_seed_makes_training_reproducible(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.training.loop import set_seed
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    weights = []
    for run in range(2):
        set_seed(7)
        model = build_model(x.shape[1])
        _fit(model, x, y, tmp_path / str(run), epochs=1)
        weights.append(model.head.weight.detach().numpy().copy())
    np.testing.assert_allclose(weights[0], weights[1])


def test_fit_rejects_mismatched_lengths(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    with pytest.raises(ValueError, match="rows but y has"):
        _fit(build_model(x.shape[1]), x, y[:-1], tmp_path)


def test_fit_writes_a_checkpoint_and_an_appending_log(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    """Two stages share one log, CSVLogger(append=True)-style: one header, all epochs."""
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    model = build_model(x.shape[1])
    first = _fit(model, x, y, tmp_path)
    _fit(model, x, y, tmp_path)

    assert (tmp_path / "best.pt").is_file()
    rows = list(csv.reader((tmp_path / "log.csv").open(encoding="utf-8")))
    assert rows[0] == ["epoch", "learning_rate", "loss", "mae", "val_loss", "val_mae"]
    assert len(rows) == 1 + 2 * len(first.history)
    assert first.monitor == "val_loss"
    assert set(first.history_columns()) == {"loss", "mae", "val_loss", "val_mae", "learning_rate"}


def test_without_a_validation_slice_the_loop_monitors_training_loss(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    """Keras silently skipped checkpoints here; the port falls back to `loss` instead."""
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    result = _fit(build_model(x.shape[1]), x, y, tmp_path, validation_split=0.0)
    assert result.monitor == "loss"
    assert "val_loss" not in result.history_columns()
    assert (tmp_path / "best.pt").is_file()


def test_early_stopping_halts_a_plateaued_run(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    # A vanishing learning rate freezes the weights, so val_loss cannot improve.
    result = _fit(
        build_model(x.shape[1]),
        x,
        y,
        tmp_path,
        epochs=20,
        learning_rate=1e-30,
        min_lr=0.0,
        early_stopping_patience=2,
    )
    assert result.stopped_early
    assert len(result.history) == 3, "best at epoch 0, then two epochs of patience"


def test_fit_leaves_the_best_weights_loaded(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.training.loop import load_checkpoint
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    model = build_model(x.shape[1])
    _fit(model, x, y, tmp_path, epochs=3)

    restored = load_checkpoint(tmp_path / "best.pt")
    for name, tensor in model.state_dict().items():
        np.testing.assert_array_equal(tensor.numpy(), restored.state_dict()[name].numpy())


def test_on_epoch_is_called_once_per_epoch(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.training.loop import fit
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    seen: list[int] = []
    fit(
        build_model(x.shape[1]),
        x,
        y,
        TrainConfig(epochs=3, batch_size=16, early_stopping_patience=10),
        checkpoint_path=tmp_path / "best.pt",
        log_path=tmp_path / "log.csv",
        on_epoch=lambda m: seen.append(m.epoch),
        verbose=0,
    )
    assert seen == [0, 1, 2]


def test_a_non_finite_loss_is_a_training_error(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    y = y.copy()
    y[0] = np.nan
    with pytest.raises(TrainingError, match="non-finite"):
        _fit(build_model(x.shape[1]), x, y, tmp_path, validation_split=0.0)


def test_sample_weights_change_the_reported_loss(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.training.loop import fit, set_seed
    from mlpp.training.model import build_model

    x, y = _xy(frame, column_cfg)
    losses = []
    for weight in (1.0, 3.0):
        set_seed(0)
        result = fit(
            build_model(x.shape[1]),
            x,
            y,
            TrainConfig(epochs=1, batch_size=16, validation_split=0.0),
            checkpoint_path=tmp_path / f"{weight}.pt",
            log_path=tmp_path / f"{weight}.csv",
            sample_weight=np.full(len(x), weight, dtype=np.float32),
            verbose=0,
        )
        losses.append(result.history[0].loss)
    assert losses[1] > losses[0]


def test_active_sample_weights_are_disabled_by_default(
    frame: pd.DataFrame, column_cfg: ColumnConfig
) -> None:
    from mlpp.training.loop import active_sample_weights

    pre = Preprocessor(column_cfg).fit(frame)
    x, _ = pre.transform(frame)
    assert active_sample_weights(x, pre, TrainConfig()) is None


def test_active_sample_weights_track_the_right_column(
    frame: pd.DataFrame, column_cfg: ColumnConfig
) -> None:
    from mlpp.training.loop import active_sample_weights

    pre = Preprocessor(column_cfg).fit(frame)
    x, _ = pre.transform(frame)
    weights = active_sample_weights(x, pre, TrainConfig(active_weight_alpha=0.5))

    assert weights is not None
    scaled_active = x[:, pre.index_of("comp_active"), 0]
    np.testing.assert_allclose(weights, 1.0 + 0.5 * (scaled_active != 0))


def test_active_sample_weights_none_when_column_absent(
    frame: pd.DataFrame, column_cfg: ColumnConfig
) -> None:
    from mlpp.training.loop import active_sample_weights

    pre = Preprocessor(column_cfg).fit(frame)
    x, _ = pre.transform(frame)
    cfg = TrainConfig(active_weight_alpha=0.5, active_column="does_not_exist")
    assert active_sample_weights(x, pre, cfg) is None


def test_load_checkpoint_rejects_a_missing_or_corrupt_file(tmp_path: Path) -> None:
    from mlpp.training.loop import load_checkpoint

    with pytest.raises(ArtifactError, match="not found"):
        load_checkpoint(tmp_path / "absent.pt")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(ArtifactError, match="cannot load checkpoint"):
        load_checkpoint(corrupt)


def test_an_unavailable_device_is_reported_not_ignored() -> None:
    import torch

    from mlpp.training.loop import resolve_device

    if torch.cuda.is_available():
        pytest.skip("CUDA is available here")
    with pytest.raises(TrainingError, match="CUDA is not available"):
        resolve_device("cuda")
    assert resolve_device("auto").type == "cpu"


# --- ONNX export ------------------------------------------------------------------------


def test_export_matches_pytorch_through_onnx_runtime(
    frame: pd.DataFrame, column_cfg: ColumnConfig, tmp_path: Path
) -> None:
    from mlpp.inference.onnx_model import OnnxModel
    from mlpp.training.export import ONNX_INPUT, ONNX_OPSET, ONNX_OUTPUT, export_onnx
    from mlpp.training.loop import predict_scaled
    from mlpp.training.model import build_model, count_parameters

    x, y = _xy(frame, column_cfg)
    model = build_model(x.shape[1])
    _fit(model, x, y, tmp_path, epochs=1)
    path = export_onnx(model, tmp_path / "model.onnx", sample=x)

    onnx_model = OnnxModel.from_path(path)
    for rows in (1, 7, len(x)):  # the batch axis must be dynamic, including size 1
        np.testing.assert_allclose(
            onnx_model.predict(x[:rows]), predict_scaled(model, x[:rows]), atol=1e-5
        )

    description = onnx_model.describe()
    assert (description.input_name, description.output_name) == (ONNX_INPUT, ONNX_OUTPUT)
    assert description.input_shape[1:] == (x.shape[1], 1)
    assert description.opset == ONNX_OPSET
    assert description.n_parameters == count_parameters(model)


def test_parity_check_rejects_a_graph_that_is_not_the_model(tmp_path: Path) -> None:
    from mlpp.training.export import export_onnx, verify_parity
    from mlpp.training.loop import set_seed
    from mlpp.training.model import build_model

    set_seed(1)
    path = export_onnx(build_model(3), tmp_path / "a.onnx")
    set_seed(2)
    impostor = build_model(3).eval()
    sample = np.random.default_rng(0).standard_normal((16, 3, 1)).astype(np.float32)
    with pytest.raises(ArtifactError, match="not a faithful copy"):
        verify_parity(impostor, path, sample, atol=1e-6)

"""Shared fixtures. Nothing here imports torch — see tests marked `slow` for that."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlpp.config import ColumnConfig, DataConfig, EvalConfig, PipelineConfig, TrainConfig

INPUT_COLUMNS = ("feature_01", "feature_02", "comp_active")
OUTPUT_COLUMN = "target"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def frame(rng: np.random.Generator) -> pd.DataFrame:
    """A small, well-formed training frame matching INPUT_COLUMNS."""
    n = 64
    return pd.DataFrame(
        {
            "feature_01": rng.normal(size=n),
            "feature_02": rng.normal(size=n),
            "comp_active": rng.integers(0, 2, size=n),
            OUTPUT_COLUMN: rng.normal(size=n),
        }
    )


@pytest.fixture
def column_cfg() -> ColumnConfig:
    """The column contract on its own — all a session reader needs."""
    return ColumnConfig(input_columns=INPUT_COLUMNS, output_column=OUTPUT_COLUMN)


@pytest.fixture
def data_cfg(tmp_path: Path, column_cfg: ColumnConfig) -> DataConfig:
    for name in ("TRAIN", "TEST", "OUTPUTS"):
        (tmp_path / name).mkdir()
    return DataConfig(
        train_dir=tmp_path / "TRAIN",
        test_dir=tmp_path / "TEST",
        output_dir=tmp_path / "OUTPUTS",
        columns=column_cfg,
    )


@pytest.fixture
def pipeline_cfg(data_cfg: DataConfig, frame: pd.DataFrame) -> PipelineConfig:
    """A runnable config with one train CSV and one test CSV already on disk."""
    frame.to_csv(data_cfg.train_dir / "train_01.csv", index=False)
    frame.to_csv(data_cfg.test_dir / "test_A.csv", index=False)
    return PipelineConfig(
        data=data_cfg,
        train=TrainConfig(epochs=1, batch_size=16, validation_split=0.2),
        eval=EvalConfig(rolling_window=4, write_plots=False),
    )


def write_linear_onnx(
    path: Path,
    weights: Sequence[float],
    *,
    bias: float = 0.0,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """A real ONNX graph computing `x[:, :, 0] @ weights + bias`, built without torch.

    Lets tests exercise genuine ONNX Runtime execution in the fast suite: the graph has
    the exported model's exact I/O contract — `features (batch, n, 1)` to
    `prediction (batch, 1)` — and a predictable output.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n = len(weights)
    graph = helper.make_graph(
        [
            helper.make_node("Squeeze", ["features", "axes"], ["flat"]),
            helper.make_node("MatMul", ["flat", "w"], ["product"]),
            helper.make_node("Add", ["product", "b"], ["prediction"]),
        ],
        "linear",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", n, 1])],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, ["batch", 1])],
        [
            numpy_helper.from_array(np.asarray(weights, dtype=np.float32).reshape(n, 1), "w"),
            numpy_helper.from_array(np.asarray([bias], dtype=np.float32), "b"),
            numpy_helper.from_array(np.asarray([2], dtype=np.int64), "axes"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="mlpp-tests"
    )
    model.ir_version = 10  # within every supported onnxruntime's range
    if metadata:
        helper.set_model_props(model, dict(metadata))
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def linear_onnx() -> Callable[..., Path]:
    return write_linear_onnx

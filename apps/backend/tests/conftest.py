"""Shared fixtures. Nothing here imports TensorFlow — see tests marked `slow` for that."""

from __future__ import annotations

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

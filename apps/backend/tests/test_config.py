from __future__ import annotations

from pathlib import Path

import pytest

from mlpp.config import ColumnConfig, DataConfig, EvalConfig, PipelineConfig, TrainConfig


def test_for_repo_derives_standard_directories(tmp_path: Path) -> None:
    cfg = PipelineConfig.for_repo(tmp_path)
    assert cfg.data.train_dir == tmp_path / "TRAIN"
    assert cfg.data.test_dir == tmp_path / "TEST"
    assert cfg.data.output_dir == tmp_path / "OUTPUTS"


def test_config_is_immutable(column_cfg: ColumnConfig) -> None:
    with pytest.raises(AttributeError):
        column_cfg.output_column = "other"  # type: ignore[misc]


def test_data_config_is_immutable(data_cfg: DataConfig) -> None:
    with pytest.raises(AttributeError):
        data_cfg.train_dir = Path("/elsewhere")  # type: ignore[misc]


def test_data_config_defaults_its_column_contract(tmp_path: Path) -> None:
    """`DataConfig` composes the column contract; omitting it yields the defaults."""
    cfg = DataConfig(train_dir=tmp_path, test_dir=tmp_path, output_dir=tmp_path)
    assert cfg.columns == ColumnConfig()


def test_categorical_columns_must_be_inputs() -> None:
    with pytest.raises(ValueError, match="not present in input_columns"):
        ColumnConfig(input_columns=("a", "b"), categorical_columns=("c",))


def test_output_column_may_not_be_an_input() -> None:
    with pytest.raises(ValueError, match="must not also be an input column"):
        ColumnConfig(input_columns=("a", "target"), output_column="target")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epochs": 0}, "epochs must be >= 1"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"validation_split": 1.0}, r"validation_split must be in \[0, 1\)"),
        ({"active_weight_alpha": -0.5}, "active_weight_alpha must be >= 0"),
    ],
)
def test_train_config_rejects_invalid_values(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TrainConfig(**kwargs)  # type: ignore[arg-type]


def test_eval_config_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="rolling_window must be >= 1"):
        EvalConfig(rolling_window=0)


def test_with_train_returns_a_modified_copy(data_cfg: DataConfig) -> None:
    cfg = PipelineConfig(data=data_cfg)
    faster = cfg.with_train(epochs=1)
    assert faster.train.epochs == 1
    assert cfg.train.epochs == 5, "original config must not be mutated"
    assert faster.data is cfg.data

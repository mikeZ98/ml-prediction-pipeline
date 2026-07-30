"""Typed configuration. Replaces the notebook's module-level globals.

Config is immutable and passed explicitly, so a test can build a tiny variant
without mutating shared state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

Loss = Literal["mse", "huber"]

DEFAULT_INPUT_COLUMNS: tuple[str, ...] = (
    "feature_01",
    "feature_02",
    "feature_03",
    "feature_04",
    "feature_05",
    "feature_06",
    "feature_07",
    "feature_08",
    "feature_09",
    "feature_10",
    "feature_11",
    "feature_12",
    "comp_active",
)
DEFAULT_OUTPUT_COLUMN = "target"
DEFAULT_ACTIVE_COLUMN = "comp_active"


@dataclass(frozen=True, slots=True)
class ColumnConfig:
    """Which columns matter — the contract, independent of where the data lives.

    Split out of `DataConfig` so a reader that scores an existing session can
    supply the column contract without inventing train/test/output directories it
    has no use for. `preprocess` and `session` depend on this, never on the paths.
    """

    input_columns: tuple[str, ...] = DEFAULT_INPUT_COLUMNS
    output_column: str = DEFAULT_OUTPUT_COLUMN
    categorical_columns: tuple[str, ...] = ()
    #: True -> a missing input column raises SchemaError; False -> fill it with 0.0.
    strict_schema: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.categorical_columns) - set(self.input_columns)
        if unknown:
            msg = f"categorical_columns not present in input_columns: {sorted(unknown)}"
            raise ValueError(msg)
        if self.output_column in self.input_columns:
            msg = f"output_column {self.output_column!r} must not also be an input column"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the data lives, plus the column contract it follows."""

    train_dir: Path
    test_dir: Path
    output_dir: Path
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    #: Explicit test filenames (relative to `test_dir`); empty means "glob everything".
    test_files: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Model and fit hyperparameters."""

    epochs: int = 5
    batch_size: int = 128
    seed: int = 50
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    loss: Loss = "mse"
    use_onehot: bool = True
    #: Column whose non-zero value marks an "active" sample.
    active_column: str | None = DEFAULT_ACTIVE_COLUMN
    #: >0 up-weights active samples as `1 + alpha`; 0 disables weighting.
    active_weight_alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError(f"validation_split must be in [0, 1), got {self.validation_split}")
        if self.active_weight_alpha < 0.0:
            raise ValueError(f"active_weight_alpha must be >= 0, got {self.active_weight_alpha}")


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Evaluation and plotting knobs."""

    rolling_window: int = 500
    write_plots: bool = True

    def __post_init__(self) -> None:
        if self.rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1, got {self.rolling_window}")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Everything `run_pipeline` needs."""

    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def for_repo(cls, root: Path) -> PipelineConfig:
        """Default config for the repo layout: TRAIN/, TEST/, OUTPUTS/ at `root`."""
        return cls(
            data=DataConfig(
                train_dir=root / "TRAIN",
                test_dir=root / "TEST",
                output_dir=root / "OUTPUTS",
            )
        )

    def with_train(self, **changes: object) -> PipelineConfig:
        """Copy with overridden training fields."""
        return replace(self, train=replace(self.train, **changes))  # type: ignore[arg-type]

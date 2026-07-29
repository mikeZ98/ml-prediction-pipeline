"""Training loop wiring: seeding, callbacks, sample weights."""

from __future__ import annotations

import random
from pathlib import Path

import keras
import numpy as np

from mlpp.config import TrainConfig
from mlpp.preprocess import Preprocessor

BEST_MODEL_FILE = "best_model.keras"
TRAINING_LOG_FILE = "training_log.csv"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Keras RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def active_sample_weights(x: np.ndarray, pre: Preprocessor, cfg: TrainConfig) -> np.ndarray | None:
    """Weight active samples as `1 + alpha`, everything else as 1.

    Returns None when weighting is off or the active column is not a model feature.
    The notebook indexed X by the *config* column order, but X's axis 1 follows the
    fitted feature order — so it read the wrong column whenever the two diverged.
    `Preprocessor.index_of` is the authoritative lookup.
    """
    if cfg.active_weight_alpha <= 0.0 or cfg.active_column is None:
        return None
    try:
        idx = pre.index_of(cfg.active_column)
    except KeyError:
        return None
    active = (x[:, idx, 0] != 0).astype(np.float32)
    return np.asarray(1.0 + cfg.active_weight_alpha * active, dtype=np.float32)


def build_callbacks(out_dir: Path, cfg: TrainConfig) -> list[keras.callbacks.Callback]:
    """Early stopping, LR plateau decay, best-checkpoint and CSV logging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=cfg.reduce_lr_patience, factor=0.5, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            str(out_dir / BEST_MODEL_FILE), monitor="val_loss", mode="min", save_best_only=True
        ),
        keras.callbacks.CSVLogger(str(out_dir / TRAINING_LOG_FILE), append=True),
    ]


def train(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    cfg: TrainConfig,
    sample_weight: np.ndarray | None = None,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Fit `model` on one dataset, writing checkpoints and logs into `out_dir`."""
    if len(x) != len(y):
        raise ValueError(f"X has {len(x)} rows but y has {len(y)}")
    return model.fit(
        x,
        y,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        callbacks=build_callbacks(out_dir, cfg),
        sample_weight=sample_weight,
        verbose=verbose,
    )

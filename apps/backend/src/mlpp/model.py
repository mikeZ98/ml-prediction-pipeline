"""Keras 3 model definition: Conv1D stack -> stacked Bi-GRU -> linear head."""

from __future__ import annotations

import contextlib

import keras
from keras import layers

from mlpp.config import TrainConfig


def build_model(n_features: int, cfg: TrainConfig) -> keras.Model:
    """Compile the CNN+Bi-GRU regressor for inputs of shape (n_features, 1)."""
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")

    model = keras.Sequential(
        [
            # Explicit Input layer: passing `input_shape=` to the first layer is
            # deprecated in Keras 3.
            layers.Input(shape=(n_features, 1), name="features"),
            layers.Conv1D(64, 1, activation="relu"),
            layers.Conv1D(128, 1, activation="relu"),
            layers.Dropout(0.2),
            layers.Conv1D(64, 1, activation="relu"),
            layers.Dropout(0.2),
            layers.Bidirectional(
                layers.GRU(
                    128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(1e-3)
                )
            ),
            layers.Dropout(0.3),
            layers.Bidirectional(
                layers.GRU(
                    64, return_sequences=False, kernel_regularizer=keras.regularizers.l2(1e-3)
                )
            ),
            layers.Dropout(0.3),
            layers.Dense(1, activation="linear"),
        ],
        name="cnn_bigru_regressor",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="mse" if cfg.loss == "mse" else keras.losses.Huber(),
        metrics=["mae", "mape"],
    )
    return model


def configure_gpu_memory_growth() -> int:
    """Enable incremental GPU memory allocation. Returns the GPU count (0 if none)."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Already-initialised devices reject the change; that is not fatal.
        with contextlib.suppress(RuntimeError):
            tf.config.experimental.set_memory_growth(gpu, True)
    return len(gpus)

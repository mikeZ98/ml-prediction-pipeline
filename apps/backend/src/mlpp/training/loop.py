"""The PyTorch fit loop: seeding, batching, sample weights, early stopping, checkpoints.

It reproduces the four Keras callbacks the pipeline relied on, explicitly:

- **EarlyStopping** on the monitored loss, patience `early_stopping_patience`, with the
  best weights restored when training ends;
- **ReduceLROnPlateau** on the same signal, factor 0.5, floor `min_lr`;
- **ModelCheckpoint(save_best_only)** — the best weights are written to
  `checkpoint_path` every time the monitor improves;
- **CSVLogger(append=True)** — one row per epoch appended to `log_path`, so a
  multi-stage run accumulates a single log.

Validation follows Keras' `validation_split`: the *last* fraction of the rows, taken
before shuffling. The monitor is `val_loss` when there is a validation slice, else
`loss`. Keras silently skipped checkpoints when `val_loss` was absent; here the
monitor falls back instead.

Reported `loss` / `val_loss` include the L2 penalty, as Keras' do. `on_epoch` is the
hook experiment tracking attaches to — this module knows nothing about MLflow.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import pickle
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mlpp.config import Device, TrainConfig
from mlpp.errors import ArtifactError, TrainingError
from mlpp.preprocess import Preprocessor
from mlpp.training.model import CnnBiGruRegressor, build_model, make_loss

log = logging.getLogger(__name__)

_LOG_COLUMNS = ("epoch", "learning_rate", "loss", "mae", "val_loss", "val_mae")
_PREDICT_BATCH = 4096


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """One epoch's numbers. `epoch` is 0-based within a single `fit` call."""

    epoch: int
    learning_rate: float
    loss: float
    mae: float
    val_loss: float | None
    val_mae: float | None

    def as_dict(self) -> dict[str, float]:
        """Metric name -> value, omitting the validation pair when there was none."""
        out = {"learning_rate": self.learning_rate, "loss": self.loss, "mae": self.mae}
        if self.val_loss is not None and self.val_mae is not None:
            out |= {"val_loss": self.val_loss, "val_mae": self.val_mae}
        return out


EpochCallback = Callable[[EpochMetrics], None]


@dataclass(frozen=True, slots=True)
class FitResult:
    history: tuple[EpochMetrics, ...]
    monitor: str
    best_epoch: int
    best_score: float
    stopped_early: bool

    def history_columns(self) -> dict[str, list[float]]:
        """Per-epoch series keyed like Keras' `History.history` (`loss`, `val_mae`, …)."""
        columns: dict[str, list[float]] = {}
        for metrics in self.history:
            for key, value in metrics.as_dict().items():
                columns.setdefault(key, []).append(value)
        return columns


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: Device) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise TrainingError("device 'cuda' requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise TrainingError("device 'mps' requested but MPS is not available")
    return torch.device(name)


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


def fit(
    model: CnnBiGruRegressor,
    x: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    *,
    checkpoint_path: Path,
    log_path: Path,
    sample_weight: np.ndarray | None = None,
    on_epoch: EpochCallback | None = None,
    verbose: int = 1,
) -> FitResult:
    """Fit `model` in place on one dataset; the best weights are loaded when it returns.

    Raises ValueError on mismatched lengths and TrainingError when the loss goes
    non-finite (the Keras `TerminateOnNaN` case, made loud).
    """
    if len(x) != len(y):
        raise ValueError(f"X has {len(x)} rows but y has {len(y)}")
    if sample_weight is not None and len(sample_weight) != len(x):
        raise ValueError(f"X has {len(x)} rows but sample_weight has {len(sample_weight)}")

    device = resolve_device(cfg.device)
    model.to(device)
    tensors = _to_tensors(x, y, sample_weight, device)
    train_set, val_set = _split(tensors, cfg.validation_split)
    monitor = "val_loss" if val_set is not None else "loss"

    loss_fn = make_loss(cfg.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.reduce_lr_patience, min_lr=cfg.min_lr
    )
    shuffler = torch.Generator().manual_seed(cfg.seed)

    history: list[EpochMetrics] = []
    best_score, best_epoch, wait = math.inf, -1, 0
    best_state: dict[str, Any] | None = None
    stopped_early = False

    for epoch in range(cfg.epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        loss, mae = _train_epoch(model, train_set, loss_fn, optimizer, cfg.batch_size, shuffler)
        val_loss, val_mae = (None, None)
        if val_set is not None:
            val_loss, val_mae = _evaluate(model, val_set, loss_fn, cfg.batch_size)

        metrics = EpochMetrics(epoch, learning_rate, loss, mae, val_loss, val_mae)
        history.append(metrics)
        _append_log(log_path, metrics)
        if on_epoch is not None:
            on_epoch(metrics)
        if verbose:
            log.info("epoch %d/%d %s", epoch + 1, cfg.epochs, _format(metrics))

        score = val_loss if val_loss is not None else loss
        if not math.isfinite(score):
            raise TrainingError(
                f"{monitor} became non-finite ({score}) at epoch {epoch}; "
                "lower the learning rate or check the inputs for extreme values"
            )
        scheduler.step(score)
        if score < best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, checkpoint_path, epoch=epoch, monitor=monitor, score=score)
        else:
            wait += 1
            if wait >= cfg.early_stopping_patience:
                stopped_early = True
                if verbose:
                    log.info("early stopping at epoch %d (best %d)", epoch + 1, best_epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(
        history=tuple(history),
        monitor=monitor,
        best_epoch=best_epoch,
        best_score=best_score,
        stopped_early=stopped_early,
    )


def predict_scaled(
    model: nn.Module, x: np.ndarray, *, batch_size: int = _PREDICT_BATCH
) -> np.ndarray:
    """Eval-mode forward pass in batches; returns scaled predictions shaped `(rows,)`."""
    device = next(model.parameters()).device
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(model(batch).reshape(-1).cpu().numpy())
    if not outputs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32)


def save_checkpoint(model: CnnBiGruRegressor, path: Path, **extra: object) -> Path:
    """Write the state dict plus what is needed to rebuild the module around it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"n_features": model.n_features, "state_dict": state, **extra}, path)
    return path


def load_checkpoint(path: Path) -> CnnBiGruRegressor:
    """Rebuild a model from `save_checkpoint` output. `weights_only` — never unpickles code."""
    if not path.is_file():
        raise ArtifactError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(int(payload["n_features"]))
        model.load_state_dict(payload["state_dict"])
    except (KeyError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise ArtifactError(f"cannot load checkpoint {path}: {exc}") from exc
    return model.eval()


# --- internals --------------------------------------------------------------------------

_Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _to_tensors(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, device: torch.device
) -> _Batch:
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=device)
    wt = (
        torch.ones(len(x), device=device)
        if weights is None
        else torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device)
    )
    return xt, yt, wt


def _split(tensors: _Batch, validation_split: float) -> tuple[_Batch, _Batch | None]:
    """Keras semantics: the trailing `validation_split` fraction, before any shuffling."""
    n = len(tensors[0])
    split_at = math.ceil(n * (1.0 - validation_split))
    if split_at < 1:
        raise ValueError(f"validation_split={validation_split} leaves no training rows of {n}")
    if split_at >= n:
        return tensors, None
    head = (tensors[0][:split_at], tensors[1][:split_at], tensors[2][:split_at])
    tail = (tensors[0][split_at:], tensors[1][split_at:], tensors[2][split_at:])
    return head, tail


def _batch_loss(
    model: CnnBiGruRegressor,
    loss_fn: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    wb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total loss, weighted |error| sum, prediction) for one batch.

    The data term is Keras' `sum_over_batch_size`: `sum(loss_i * w_i) / batch`.
    """
    pred = model(xb)
    per_sample = loss_fn(pred, yb).mean(dim=1)
    total = (per_sample * wb).sum() / len(xb) + model.l2_penalty()
    abs_error = ((pred - yb).abs().mean(dim=1) * wb).sum()
    return total, abs_error, pred


def _train_epoch(
    model: CnnBiGruRegressor,
    data: _Batch,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    shuffler: torch.Generator,
) -> tuple[float, float]:
    model.train()
    x, y, w = data
    order = torch.randperm(len(x), generator=shuffler).to(x.device)
    loss_sum, abs_sum = 0.0, 0.0
    for start in range(0, len(x), batch_size):
        idx = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        total, abs_error, _ = _batch_loss(model, loss_fn, x[idx], y[idx], w[idx])
        total.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        loss_sum += float(total.detach()) * len(idx)
        abs_sum += float(abs_error.detach())
    return loss_sum / len(x), abs_sum / float(w.sum())


def _evaluate(
    model: CnnBiGruRegressor, data: _Batch, loss_fn: nn.Module, batch_size: int
) -> tuple[float, float]:
    model.eval()
    x, y, w = data
    loss_sum, abs_sum = 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = start + batch_size
            total, abs_error, _ = _batch_loss(
                model, loss_fn, x[start:stop], y[start:stop], w[start:stop]
            )
            loss_sum += float(total) * len(x[start:stop])
            abs_sum += float(abs_error)
    return loss_sum / len(x), abs_sum / float(w.sum())


def _append_log(path: Path, metrics: EpochMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(_LOG_COLUMNS)
        writer.writerow(
            [
                metrics.epoch,
                metrics.learning_rate,
                metrics.loss,
                metrics.mae,
                "" if metrics.val_loss is None else metrics.val_loss,
                "" if metrics.val_mae is None else metrics.val_mae,
            ]
        )


def _format(metrics: EpochMetrics) -> str:
    return " ".join(f"{key}={value:.4g}" for key, value in metrics.as_dict().items())

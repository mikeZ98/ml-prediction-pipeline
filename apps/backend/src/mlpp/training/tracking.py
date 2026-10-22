"""MLflow experiment tracking behind a narrow `Tracker` protocol.

The pipeline talks to a `Tracker`, never to MLflow directly, for two reasons:

- tracking is optional — `NullTracker` makes "off" a real object rather than an
  `if` at every call site, and keeps the library default free of a tracking store;
- MLflow is a heavy, process-global import. It is loaded only inside
  `start_tracking` when tracking is enabled, so this module imports cheaply and
  without torch.

What a run records:

- **params** — every TrainConfig/EvalConfig field plus the column contract;
- **per-epoch metrics** — `train_loss`, `train_mae`, `val_loss`, `val_mae`,
  `learning_rate`, stepped by a global epoch counter across training stages;
- **evaluation metrics** — `eval.stage_NN.<test file>.<rmse|mae|mse|r2>`, plus the
  final stage's means as `test_<metric>_mean` for comparing runs at a glance;
- **artifacts** — the whole session directory under `session/`, and the ONNX graph
  as an MLflow model under `model/` (signature + input example), optionally
  registered in the Model Registry.

`start_tracking` uses `mlflow.start_run` as a context manager, so a run that raises
is closed as FAILED rather than left RUNNING.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from mlpp.config import TrackingConfig
from mlpp.errors import TrackingError

if TYPE_CHECKING:  # the loop imports torch; only its dataclass type is needed here
    from mlpp.training.loop import EpochMetrics

#: MLflow metric keys allow alphanumerics, `_ - . /` and spaces.
_UNSAFE_KEY = re.compile(r"[^A-Za-z0-9_\-./ ]")
#: MLflow rejects parameter values longer than this.
_MAX_PARAM_LENGTH = 6000


class Tracker(Protocol):
    @property
    def run_id(self) -> str | None: ...

    def set_tags(self, tags: Mapping[str, str]) -> None: ...

    def log_params(self, params: Mapping[str, object]) -> None: ...

    def log_epoch(self, metrics: EpochMetrics, *, stage: int, step: int) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None: ...

    def log_session(self, session_dir: Path) -> None: ...

    def log_onnx_model(self, path: Path, *, input_example: np.ndarray) -> None: ...


class NullTracker:
    """Tracking switched off: every call is accepted and discarded."""

    @property
    def run_id(self) -> str | None:
        return None

    def set_tags(self, tags: Mapping[str, str]) -> None:
        return None

    def log_params(self, params: Mapping[str, object]) -> None:
        return None

    def log_epoch(self, metrics: EpochMetrics, *, stage: int, step: int) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        return None

    def log_session(self, session_dir: Path) -> None:
        return None

    def log_onnx_model(self, path: Path, *, input_example: np.ndarray) -> None:
        return None


class MlflowTracker:
    """Writes to the active MLflow run opened by `start_tracking`."""

    def __init__(self, mlflow: Any, run_id: str, cfg: TrackingConfig) -> None:
        self._mlflow = mlflow
        self._run_id = run_id
        self._cfg = cfg

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._mlflow.set_tags(dict(tags))

    def log_params(self, params: Mapping[str, object]) -> None:
        self._mlflow.log_params({key: _param_value(value) for key, value in params.items()})

    def log_epoch(self, metrics: EpochMetrics, *, stage: int, step: int) -> None:
        values = {"stage": float(stage), "learning_rate": metrics.learning_rate}
        values |= {"train_loss": metrics.loss, "train_mae": metrics.mae}
        if metrics.val_loss is not None and metrics.val_mae is not None:
            values |= {"val_loss": metrics.val_loss, "val_mae": metrics.val_mae}
        self._mlflow.log_metrics(values, step=step)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        self._mlflow.log_metrics(
            {metric_key(key): float(value) for key, value in metrics.items()}, step=step
        )

    def log_session(self, session_dir: Path) -> None:
        self._mlflow.log_artifacts(str(session_dir), artifact_path="session")

    def log_onnx_model(self, path: Path, *, input_example: np.ndarray) -> None:
        """Log the exported graph as an MLflow ONNX model, scored to derive its signature.

        The signature is inferred from what ONNX Runtime actually returns for the
        example, so the recorded contract is the served one, not an assumption.
        """
        if not self._cfg.log_model:
            return
        import onnx
        from mlflow.models import infer_signature

        from mlpp.inference.onnx_model import OnnxModel

        example = np.ascontiguousarray(input_example, dtype=np.float32)
        predictions = OnnxModel.from_path(path).predict(example).reshape(-1, 1)
        self._mlflow.onnx.log_model(
            onnx.load(str(path)),
            name="model",
            signature=infer_signature(example, predictions),
            input_example=example,
            registered_model_name=self._cfg.registered_model_name,
            save_as_external_data=False,
            # Matches how mlpp serves it; MLflow's default also lists CUDA.
            onnx_execution_providers=["CPUExecutionProvider"],
        )


@contextlib.contextmanager
def start_tracking(cfg: TrackingConfig, *, run_name: str) -> Iterator[Tracker]:
    """Open an MLflow run for the duration of the block, or a `NullTracker` when disabled.

    Raises TrackingError when the store cannot be reached or the experiment cannot be
    resolved — before any training starts, so a misconfigured URI fails in seconds,
    not after an hour of fitting.
    """
    if not cfg.enabled:
        yield NullTracker()
        return

    import mlflow
    from mlflow.exceptions import MlflowException

    try:
        if cfg.tracking_uri:
            _ensure_sqlite_parent(cfg.tracking_uri)
            mlflow.set_tracking_uri(cfg.tracking_uri)
        experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(
                cfg.experiment_name, artifact_location=cfg.artifact_location
            )
        )
        run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    except (MlflowException, OSError) as exc:
        raise TrackingError(
            f"cannot start an MLflow run at {mlflow.get_tracking_uri()!r}: {exc}"
        ) from exc

    # The run's own context manager marks it FAILED if the block raises.
    with run:
        yield MlflowTracker(mlflow, run.info.run_id, cfg)


def metric_key(raw: str) -> str:
    """`raw` with characters MLflow rejects in metric keys replaced by `_`."""
    return _UNSAFE_KEY.sub("_", raw)


def _param_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        value = ",".join(str(v) for v in value)
    text = str(value)
    return text if len(text) <= _MAX_PARAM_LENGTH else text[: _MAX_PARAM_LENGTH - 3] + "..."


def _ensure_sqlite_parent(uri: str) -> None:
    """SQLite will not create the directory its database file lives in."""
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        Path(uri[len(prefix) :]).parent.mkdir(parents=True, exist_ok=True)

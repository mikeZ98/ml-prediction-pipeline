"""End-to-end orchestration: fit on each TRAIN file in turn, evaluate on every TEST file."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mlpp.config import PipelineConfig
from mlpp.data import discover_csvs, read_csv_auto
from mlpp.errors import DatasetError
from mlpp.metrics import RegressionMetrics, regression_metrics
from mlpp.preprocess import Preprocessor
from mlpp.session import (
    BEST_CHECKPOINT_FILE,
    BEST_MODEL_FILE,
    METRICS_FILE,
    ROLE_BEST_CHECKPOINT,
    ROLE_BEST_MODEL,
    ROLE_METRICS,
    ROLE_PREDICTION_ANALYSIS,
    ROLE_STAGE_HISTORY,
    ROLE_STAGE_MODEL,
    ROLE_TRAINING_CURVES,
    ROLE_TRAINING_LOG,
    SCHEMA_VERSION,
    TRAINING_LOG_FILE,
    FeatureContract,
    SessionWriter,
    prediction_analysis_filename,
    stage_history_filename,
    stage_model_filename,
    training_curves_filename,
    write_fitted_state,
)
from mlpp.training.export import export_onnx
from mlpp.training.loop import (
    EpochCallback,
    EpochMetrics,
    FitResult,
    active_sample_weights,
    fit,
    predict_scaled,
    save_checkpoint,
    set_seed,
)
from mlpp.training.model import CnnBiGruRegressor, build_model, count_parameters
from mlpp.training.tracking import Tracker, start_tracking

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """One (train stage, test file) evaluation result."""

    train_file: str
    test_file: str
    metrics: RegressionMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "train_file": self.train_file,
            "test_file": self.test_file,
            **self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    session_dir: Path
    rows: tuple[EvaluationRow, ...]
    #: The MLflow run that recorded this session, when tracking was enabled.
    run_id: str | None = None


@dataclass(slots=True)
class _Trained:
    """What the stage loop leaves behind for export, persistence and tracking."""

    model: CnnBiGruRegressor | None = None
    last_x: np.ndarray | None = None
    rows: list[EvaluationRow] = field(default_factory=list)
    epochs_done: int = 0


def run_pipeline(cfg: PipelineConfig, verbose: int = 1) -> PipelineResult:
    """Train sequentially over the TRAIN CSVs, evaluating against TEST after each stage.

    With `cfg.tracking.enabled`, the whole run is recorded as one MLflow run named
    after the session directory: params up front, metrics per epoch and per
    evaluation, then the session artifacts and the ONNX model once they exist.

    Raises DatasetError when TRAIN/ holds no CSVs — there is nothing to fit — and
    TrackingError when tracking is enabled but the store cannot be reached.
    """
    set_seed(cfg.train.seed)
    train_files = discover_csvs(cfg.data.train_dir)
    if not train_files:
        raise DatasetError(f"no CSV files in {cfg.data.train_dir}")
    test_files = discover_csvs(cfg.data.test_dir, cfg.data.test_files)
    if not test_files:
        log.warning("no CSV files in %s — training without evaluation", cfg.data.test_dir)

    # The directory must exist before training, but the feature contract does not
    # exist until the first fit and the inventory grows across stages — hence
    # create now, set_features at fit, flush at the end.
    writer = SessionWriter.create(cfg.data.output_dir)
    session_dir = writer.session_dir
    log.info("session directory: %s", session_dir)

    with start_tracking(cfg.tracking, run_name=session_dir.name) as tracker:
        tracker.set_tags(
            {"mlpp.session": session_dir.name, "mlpp.schema_version": str(SCHEMA_VERSION)}
        )
        tracker.log_params(_run_params(cfg, train_files, test_files))

        trained = _train_stages(cfg, writer, train_files, test_files, tracker, verbose)
        onnx_path = _persist(writer, trained, cfg)

        if trained.rows:
            tracker.log_metrics(_final_stage_means(trained.rows))
        if writer.has_features:
            tracker.log_session(session_dir)
        if onnx_path is not None and trained.last_x is not None:
            tracker.log_onnx_model(onnx_path, input_example=trained.last_x[:5])

    return PipelineResult(session_dir=session_dir, rows=tuple(trained.rows), run_id=tracker.run_id)


def _train_stages(
    cfg: PipelineConfig,
    writer: SessionWriter,
    train_files: list[Path],
    test_files: list[Path],
    tracker: Tracker,
    verbose: int,
) -> _Trained:
    pre = Preprocessor(cfg.data.columns, use_onehot=cfg.train.use_onehot)
    trained = _Trained()

    for stage, train_path in enumerate(train_files, start=1):
        frame = read_csv_auto(train_path)
        # Fit only on the first file; later stages must reuse that feature space.
        if not pre.is_fitted:
            pre.fit(frame)
            writer.set_features(_contract_from(pre))
            tracker.log_params({"data.n_features": pre.n_features})
        x_train, y_train = pre.transform(frame)
        if y_train is None:
            log.warning(
                "skipping %s: no %r column", train_path.name, cfg.data.columns.output_column
            )
            continue

        if trained.model is None:
            trained.model = build_model(pre.n_features)
            tracker.log_params({"model.n_parameters": count_parameters(trained.model)})
            if verbose:
                log.info(
                    "model (%d parameters):\n%s", count_parameters(trained.model), trained.model
                )

        log.info(
            "stage %d/%d: fitting on %s %s", stage, len(train_files), train_path.name, x_train.shape
        )
        result = fit(
            trained.model,
            x_train,
            y_train,
            cfg.train,
            checkpoint_path=writer.path_for(BEST_CHECKPOINT_FILE),
            log_path=writer.path_for(TRAINING_LOG_FILE),
            sample_weight=active_sample_weights(x_train, pre, cfg.train),
            on_epoch=_epoch_logger(tracker, stage, offset=trained.epochs_done),
            verbose=verbose,
        )
        trained.epochs_done += len(result.history)
        # Written by the fit loop as it runs, so record them rather than name them.
        writer.register(ROLE_BEST_CHECKPOINT, BEST_CHECKPOINT_FILE)
        writer.register(ROLE_TRAINING_LOG, TRAINING_LOG_FILE)
        _save_stage_artifacts(
            trained.model, pre, result, writer, stage, write_plots=cfg.eval.write_plots
        )
        stage_rows = _evaluate(trained.model, pre, cfg, test_files, writer, stage, train_path.name)
        tracker.log_metrics(_evaluation_metrics(stage, stage_rows), step=stage)
        trained.rows.extend(stage_rows)
        trained.last_x = x_train
    return trained


def _persist(writer: SessionWriter, trained: _Trained, cfg: PipelineConfig) -> Path | None:
    """Export the ONNX model, write the metrics table and flush the manifest.

    Returns the ONNX path, or None when no stage produced a model.
    """
    onnx_path = None
    if trained.model is not None and trained.last_x is not None:
        # Exported once, from the weights the last stage settled on — the same model
        # the evaluation rows describe. Verified against ONNX Runtime on real rows.
        onnx_path = export_onnx(
            trained.model,
            writer.register(ROLE_BEST_MODEL, BEST_MODEL_FILE),
            sample=trained.last_x[:256],
        )
        log.info("ONNX model written to %s", onnx_path)

    if trained.rows:
        metrics_path = writer.register(ROLE_METRICS, METRICS_FILE)
        pd.DataFrame([r.as_dict() for r in trained.rows]).to_csv(metrics_path, index=False)
        log.info("metrics written to %s", metrics_path)

    if writer.has_features:
        log.info("manifest written to %s", writer.flush())
    else:
        log.warning("no file in %s could be fitted — no manifest written", cfg.data.train_dir)
    return onnx_path


def _epoch_logger(tracker: Tracker, stage: int, *, offset: int) -> EpochCallback:
    """Per-epoch callback stepping a global epoch counter across stages."""

    def log_epoch(metrics: EpochMetrics) -> None:
        tracker.log_epoch(metrics, stage=stage, step=offset + metrics.epoch)

    return log_epoch


def _run_params(
    cfg: PipelineConfig, train_files: list[Path], test_files: list[Path]
) -> dict[str, object]:
    columns = cfg.data.columns
    params: dict[str, object] = {f"train.{k}": v for k, v in asdict(cfg.train).items()}
    params |= {f"eval.{k}": v for k, v in asdict(cfg.eval).items()}
    params |= {
        "data.input_columns": columns.input_columns,
        "data.output_column": columns.output_column,
        "data.categorical_columns": columns.categorical_columns or "-",
        "data.strict_schema": columns.strict_schema,
        "data.train_files": [p.name for p in train_files],
        "data.test_files": [p.name for p in test_files] or "-",
    }
    return params


def _evaluation_metrics(stage: int, rows: list[EvaluationRow]) -> dict[str, float]:
    return {
        f"eval.stage_{stage:02d}.{Path(row.test_file).stem}.{name}": value
        for row in rows
        for name, value in row.metrics.as_dict().items()
    }


def _final_stage_means(rows: list[EvaluationRow]) -> dict[str, float]:
    """Mean test metrics of the last stage — the model that was exported."""
    last = [row for row in rows if row.train_file == rows[-1].train_file]
    names = last[0].metrics.as_dict().keys()
    return {
        f"test_{name}_mean": float(np.mean([row.metrics.as_dict()[name] for row in last]))
        for name in names
    }


def _contract_from(pre: Preprocessor) -> FeatureContract:
    """Project the fitted schema into the manifest's feature contract."""
    return FeatureContract(
        numeric_columns=pre.schema.numeric,
        categorical_columns=pre.schema.categorical,
        output_column=pre.schema.output,
        feature_names=pre.feature_names,
    )


def _save_stage_artifacts(
    model: CnnBiGruRegressor,
    pre: Preprocessor,
    result: FitResult,
    writer: SessionWriter,
    stage: int,
    *,
    write_plots: bool,
) -> None:
    """Write one stage's artifacts, registering each with the session owner.

    The old `model_iter_NN_config.json` is gone: `n_features` and `feature_names`
    are in the manifest, and duplicating them is what let the copies drift.

    Plot writing follows `eval.write_plots`, not `verbose`. Those were crossed:
    `--quiet` silently dropped the training curves while `--no-plots` left them
    on, so console noise and artifact production were the same switch.
    """
    history = result.history_columns()
    pd.DataFrame(history).to_csv(
        writer.register(ROLE_STAGE_HISTORY, stage_history_filename(stage)), index=False
    )
    write_fitted_state(writer, pre.fitted_state)
    save_checkpoint(
        model, writer.register(ROLE_STAGE_MODEL, stage_model_filename(stage)), stage=stage
    )
    if write_plots:
        from mlpp.plots import plot_training_curves

        tag = f"train_{stage:02d}"
        plot_training_curves(
            history, writer.register(ROLE_TRAINING_CURVES, training_curves_filename(tag))
        )


def _evaluate(
    model: CnnBiGruRegressor,
    pre: Preprocessor,
    cfg: PipelineConfig,
    test_files: list[Path],
    writer: SessionWriter,
    stage: int,
    train_name: str,
) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for index, test_path in enumerate(test_files, start=1):
        try:
            frame = read_csv_auto(test_path)
        except DatasetError as exc:
            log.warning("skipping %s: %s", test_path.name, exc)
            continue
        x_test, y_test = pre.transform(frame)
        if y_test is None:
            log.warning("skipping %s: no %r column", test_path.name, cfg.data.columns.output_column)
            continue

        y_pred = pre.inverse_target(predict_scaled(model, x_test))
        y_true = pre.inverse_target(y_test)
        metrics = regression_metrics(y_true, y_pred)
        rows.append(EvaluationRow(train_name, test_path.name, metrics))
        log.info(
            "%s: RMSE=%.3f MAE=%.3f R2=%.3f", test_path.name, metrics.rmse, metrics.mae, metrics.r2
        )

        if cfg.eval.write_plots:
            from mlpp.plots import plot_prediction_analysis

            tag = f"tr{stage:02d}_te{index:02d}"
            plot_prediction_analysis(
                y_true,
                y_pred,
                writer.register(ROLE_PREDICTION_ANALYSIS, prediction_analysis_filename(tag)),
                tag=tag,
                window=cfg.eval.rolling_window,
                active_mask=_active_mask(x_test, pre, cfg),
            )
    return rows


def _active_mask(x_test: np.ndarray, pre: Preprocessor, cfg: PipelineConfig) -> np.ndarray | None:
    """Boolean mask of active samples, or None when the column is not a feature."""
    if cfg.train.active_column is None:
        return None
    try:
        idx = pre.index_of(cfg.train.active_column)
    except KeyError:
        return None
    return np.asarray(x_test[:, idx, 0] != 0)

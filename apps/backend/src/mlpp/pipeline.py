"""End-to-end orchestration: fit on each TRAIN file in turn, evaluate on every TEST file."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from mlpp.artifacts import make_session_dir, save_feature_config
from mlpp.config import PipelineConfig
from mlpp.data import discover_csvs, read_csv_auto
from mlpp.errors import DatasetError
from mlpp.metrics import RegressionMetrics, regression_metrics
from mlpp.model import build_model
from mlpp.preprocess import Preprocessor
from mlpp.training import active_sample_weights, set_seed, train

log = logging.getLogger(__name__)

METRICS_FILE = "test_metrics.csv"


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


def run_pipeline(cfg: PipelineConfig, verbose: int = 1) -> PipelineResult:
    """Train sequentially over the TRAIN CSVs, evaluating against TEST after each stage.

    Raises DatasetError when TRAIN/ holds no CSVs — there is nothing to fit.
    """
    set_seed(cfg.train.seed)
    train_files = discover_csvs(cfg.data.train_dir)
    if not train_files:
        raise DatasetError(f"no CSV files in {cfg.data.train_dir}")
    test_files = discover_csvs(cfg.data.test_dir, cfg.data.test_files)
    if not test_files:
        log.warning("no CSV files in %s — training without evaluation", cfg.data.test_dir)

    session_dir = make_session_dir(cfg.data.output_dir)
    log.info("session directory: %s", session_dir)

    pre = Preprocessor(cfg.data, use_onehot=cfg.train.use_onehot)
    model = None
    rows: list[EvaluationRow] = []

    for stage, train_path in enumerate(train_files, start=1):
        frame = read_csv_auto(train_path)
        # Fit only on the first file; later stages must reuse that feature space.
        if not pre.is_fitted:
            pre.fit(frame)
            save_feature_config(pre.schema, pre.feature_names, session_dir)
        x_train, y_train = pre.transform(frame)
        if y_train is None:
            log.warning("skipping %s: no %r column", train_path.name, cfg.data.output_column)
            continue

        if model is None:
            model = build_model(pre.n_features, cfg.train)
            if verbose:
                model.summary()

        log.info(
            "stage %d/%d: fitting on %s %s", stage, len(train_files), train_path.name, x_train.shape
        )
        history = train(
            model,
            x_train,
            y_train,
            session_dir,
            cfg.train,
            sample_weight=active_sample_weights(x_train, pre, cfg.train),
            verbose=verbose,
        )
        _save_stage_artifacts(model, pre, history.history, session_dir, stage, verbose=verbose)
        rows.extend(_evaluate(model, pre, cfg, test_files, session_dir, stage, train_path.name))

    if rows:
        pd.DataFrame([r.as_dict() for r in rows]).to_csv(session_dir / METRICS_FILE, index=False)
        log.info("metrics written to %s", session_dir / METRICS_FILE)
    return PipelineResult(session_dir=session_dir, rows=tuple(rows))


def _save_stage_artifacts(
    model: keras.Model,
    pre: Preprocessor,
    history: dict[str, list[float]],
    session_dir: Path,
    stage: int,
    *,
    verbose: int,
) -> None:
    pd.DataFrame(history).to_csv(session_dir / f"history_train_{stage:02d}.csv", index=False)
    pre.save(session_dir)
    model.save(session_dir / f"model_iter_{stage:02d}.keras")
    (session_dir / f"model_iter_{stage:02d}_config.json").write_text(
        json.dumps(
            {"n_features": pre.n_features, "feature_names": list(pre.feature_names)}, indent=2
        ),
        encoding="utf-8",
    )
    if verbose:
        from mlpp.plots import plot_training_curves

        plot_training_curves(history, session_dir, tag=f"train_{stage:02d}")


def _evaluate(
    model: keras.Model,
    pre: Preprocessor,
    cfg: PipelineConfig,
    test_files: list[Path],
    session_dir: Path,
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
            log.warning("skipping %s: no %r column", test_path.name, cfg.data.output_column)
            continue

        y_pred = pre.inverse_target(model.predict(x_test, verbose=0))
        y_true = pre.inverse_target(y_test)
        metrics = regression_metrics(y_true, y_pred)
        rows.append(EvaluationRow(train_name, test_path.name, metrics))
        log.info(
            "%s: RMSE=%.3f MAE=%.3f R2=%.3f", test_path.name, metrics.rmse, metrics.mae, metrics.r2
        )

        if cfg.eval.write_plots:
            from mlpp.plots import plot_prediction_analysis

            plot_prediction_analysis(
                y_true,
                y_pred,
                session_dir,
                tag=f"tr{stage:02d}_te{index:02d}",
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

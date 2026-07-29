"""End-to-end smoke tests. Slow: these actually fit a Keras model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mlpp.config import PipelineConfig
from mlpp.errors import DatasetError
from mlpp.session import (
    MANIFEST_FILE,
    METRICS_FILE,
    ROLE_METRICS,
    ROLE_STAGE_MODEL,
    SCALER_FILE,
    read_manifest,
    stage_model_filename,
)

pytestmark = pytest.mark.slow


def test_pipeline_produces_metrics_and_artifacts(pipeline_cfg: PipelineConfig) -> None:
    from mlpp.pipeline import run_pipeline

    result = run_pipeline(pipeline_cfg, verbose=0)
    session = result.session_dir

    assert len(result.rows) == 1
    assert result.rows[0].test_file == "test_A.csv"
    for name in (MANIFEST_FILE, METRICS_FILE, stage_model_filename(1), SCALER_FILE):
        assert (session / name).is_file(), f"missing artifact {name}"

    metrics = pd.read_csv(session / METRICS_FILE)
    assert list(metrics.columns) == ["train_file", "test_file", "mse", "rmse", "mae", "r2"]


def test_pipeline_drops_the_superseded_side_cars(pipeline_cfg: PipelineConfig) -> None:
    """feature_config.json and model_iter_NN_config.json are now manifest fields."""
    from mlpp.pipeline import run_pipeline

    session = run_pipeline(pipeline_cfg, verbose=0).session_dir
    assert not (session / "feature_config.json").exists()
    assert not (session / "model_iter_01_config.json").exists()


def test_manifest_inventory_matches_files_on_disk(pipeline_cfg: PipelineConfig) -> None:
    """Every file the manifest claims must actually exist — the check that was missing."""
    from mlpp.pipeline import run_pipeline

    session = run_pipeline(pipeline_cfg, verbose=0).session_dir
    manifest = read_manifest(session)
    assert manifest.artifacts, "inventory should not be empty"
    for entry in manifest.artifacts:
        assert (session / entry.filename).is_file(), f"{entry.filename} listed but absent"
    assert manifest.filenames_for(ROLE_METRICS) == (METRICS_FILE,)
    assert manifest.filenames_for(ROLE_STAGE_MODEL) == (stage_model_filename(1),)


def test_manifest_records_the_feature_contract_once(pipeline_cfg: PipelineConfig) -> None:
    from mlpp.pipeline import run_pipeline

    result = run_pipeline(pipeline_cfg, verbose=0)
    features = read_manifest(result.session_dir).features
    assert features.output_column == pipeline_cfg.data.output_column
    assert features.feature_names, "feature names must be recorded"


def test_real_session_round_trips_through_load_session(pipeline_cfg: PipelineConfig) -> None:
    """Round trip against a directory the pipeline actually produced, not a fixture."""
    from mlpp.pipeline import run_pipeline
    from mlpp.session import load_session

    result = run_pipeline(pipeline_cfg, verbose=0)
    loaded = load_session(result.session_dir, pipeline_cfg.data)

    assert loaded.manifest.features.feature_names
    assert loaded.preprocessor.n_features == len(loaded.manifest.features.feature_names)

    frame = pd.read_csv(pipeline_cfg.data.test_dir / "test_A.csv")
    x, y = loaded.preprocessor.transform(frame)
    assert x.shape == (len(frame), loaded.preprocessor.n_features, 1)
    assert y is not None


def test_pipeline_evaluates_every_train_test_pair(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame
) -> None:
    from mlpp.pipeline import run_pipeline

    frame.to_csv(pipeline_cfg.data.train_dir / "train_02.csv", index=False)
    frame.to_csv(pipeline_cfg.data.test_dir / "test_B.csv", index=False)
    result = run_pipeline(pipeline_cfg, verbose=0)
    assert len(result.rows) == 4, "2 train stages x 2 test files"


def test_pipeline_fits_preprocessor_only_once(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame
) -> None:
    """Stage 2 must reuse stage 1's scaling, so a shifted second file stays shifted."""
    from mlpp.pipeline import run_pipeline

    shifted = frame.copy()
    shifted["feature_01"] += 50.0
    shifted.to_csv(pipeline_cfg.data.train_dir / "train_02.csv", index=False)

    result = run_pipeline(pipeline_cfg, verbose=0)
    assert (result.session_dir / "history_train_02.csv").is_file()


def test_pipeline_writes_plots_when_enabled(pipeline_cfg: PipelineConfig) -> None:
    from mlpp.pipeline import run_pipeline

    cfg = replace(pipeline_cfg, eval=replace(pipeline_cfg.eval, write_plots=True))
    session = run_pipeline(cfg, verbose=1).session_dir
    assert (session / "training_curves_train_01.png").is_file()
    assert (session / "prediction_analysis_tr01_te01.html").is_file()


def test_plots_follow_write_plots_not_verbose(pipeline_cfg: PipelineConfig) -> None:
    """These were crossed: --quiet silently dropped the training curves.

    verbose controls console noise; write_plots controls artifacts. A quiet run
    must still produce every plot.
    """
    from mlpp.pipeline import run_pipeline

    cfg = replace(pipeline_cfg, eval=replace(pipeline_cfg.eval, write_plots=True))
    session = run_pipeline(cfg, verbose=0).session_dir
    assert (session / "training_curves_train_01.png").is_file()
    assert (session / "prediction_analysis_tr01_te01.html").is_file()


def test_no_plots_suppresses_both_kinds(pipeline_cfg: PipelineConfig) -> None:
    """The converse: write_plots=False must stop curves too, even when verbose."""
    from mlpp.pipeline import run_pipeline

    cfg = replace(pipeline_cfg, eval=replace(pipeline_cfg.eval, write_plots=False))
    session = run_pipeline(cfg, verbose=1).session_dir
    assert not list(session.glob("training_curves_*.png"))
    assert not list(session.glob("prediction_analysis_*.html"))


def test_pipeline_requires_training_data(pipeline_cfg: PipelineConfig) -> None:
    """The notebook raised NameError here (`print(f(...))`); it must be a clear error."""
    from mlpp.pipeline import run_pipeline

    for csv in pipeline_cfg.data.train_dir.glob("*.csv"):
        csv.unlink()
    with pytest.raises(DatasetError, match="no CSV files"):
        run_pipeline(pipeline_cfg, verbose=0)


def test_pipeline_trains_without_test_data(pipeline_cfg: PipelineConfig) -> None:
    from mlpp.pipeline import run_pipeline

    for csv in pipeline_cfg.data.test_dir.glob("*.csv"):
        csv.unlink()
    result = run_pipeline(pipeline_cfg, verbose=0)
    assert result.rows == ()
    assert (result.session_dir / "model_iter_01.keras").is_file()


def test_cli_runs_end_to_end(pipeline_cfg: PipelineConfig, tmp_path: Path) -> None:
    from mlpp.cli import main

    exit_code = main(
        [
            "--train-dir",
            str(pipeline_cfg.data.train_dir),
            "--test-dir",
            str(pipeline_cfg.data.test_dir),
            "--output-dir",
            str(pipeline_cfg.data.output_dir),
            "--input-columns",
            *pipeline_cfg.data.input_columns,
            "--output-column",
            pipeline_cfg.data.output_column,
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--no-plots",
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert any(pipeline_cfg.data.output_dir.iterdir())


def test_cli_reports_missing_data_as_error(tmp_path: Path, capsys) -> None:
    from mlpp.cli import main

    for name in ("TRAIN", "TEST", "OUTPUTS"):
        (tmp_path / name).mkdir()
    exit_code = main(
        [
            "--train-dir",
            str(tmp_path / "TRAIN"),
            "--test-dir",
            str(tmp_path / "TEST"),
            "--output-dir",
            str(tmp_path / "OUTPUTS"),
            "--quiet",
        ]
    )
    assert exit_code == 1
    assert "no CSV files" in capsys.readouterr().err

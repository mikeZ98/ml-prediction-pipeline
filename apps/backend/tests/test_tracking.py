"""MLflow tracking. Config resolution is fast; anything that opens a run is slow."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlpp.cli import build_parser, tracking_from_args
from mlpp.config import PipelineConfig, TrackingConfig
from mlpp.errors import TrackingError, TrainingError
from mlpp.training.tracking import NullTracker, metric_key, start_tracking


@pytest.fixture
def tracking_cfg(tmp_path: Path) -> TrackingConfig:
    store = tmp_path / "mlruns"
    return TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{store / 'mlflow.db'}",
        experiment_name="mlpp-tests",
        artifact_location=(store / "artifacts").as_uri(),
    )


# --- fast: resolution and the disabled path ---------------------------------------------


def test_disabled_tracking_yields_a_null_tracker_without_importing_mlflow() -> None:
    """The library default must cost nothing. Checked in a subprocess: sys.modules is
    process-global, so an in-process check would measure test ordering instead."""
    program = textwrap.dedent("""
        import sys
        from mlpp.config import TrackingConfig
        from mlpp.training.tracking import NullTracker, start_tracking

        with start_tracking(TrackingConfig(), run_name="x") as tracker:
            assert isinstance(tracker, NullTracker)
            assert tracker.run_id is None
        for banned in ("mlflow", "torch"):
            assert banned not in sys.modules, banned + " was imported"
    """)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_null_tracker_accepts_every_call() -> None:
    tracker = NullTracker()
    tracker.set_tags({"a": "b"})
    tracker.log_params({"a": 1})
    tracker.log_metrics({"a": 1.0}, step=1)
    tracker.log_session(Path("nowhere"))
    tracker.log_onnx_model(Path("nowhere.onnx"), input_example=np.zeros((1, 1, 1)))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eval.stage_01.sample_test_A.rmse", "eval.stage_01.sample_test_A.rmse"),
        ("eval.stage_01.test (copy)#2.r2", "eval.stage_01.test _copy__2.r2"),
        ("a/b-c d", "a/b-c d"),
    ],
)
def test_metric_keys_are_sanitised(raw: str, expected: str) -> None:
    assert metric_key(raw) == expected


def test_cli_no_tracking_disables_mlflow(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--no-tracking"])
    assert tracking_from_args(args, tmp_path).enabled is False


def test_cli_prefers_the_flag_then_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5050")
    from_env = tracking_from_args(build_parser().parse_args([]), tmp_path)
    assert (from_env.enabled, from_env.tracking_uri) == (True, "http://env:5050")
    assert from_env.artifact_location is None, "a server owns its artifact location"

    flagged = build_parser().parse_args(["--tracking-uri", "http://flag:5050"])
    assert tracking_from_args(flagged, tmp_path).tracking_uri == "http://flag:5050"


def test_cli_falls_back_to_a_store_under_the_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not MLflow's default, which drops mlflow.db into the current directory."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = tracking_from_args(build_parser().parse_args([]), tmp_path)
    assert cfg.tracking_uri == f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
    assert cfg.artifact_location == (tmp_path / "mlruns" / "artifacts").as_uri()


def test_cli_maps_experiment_and_registry_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--experiment", "exp", "--register-model", "mlpp-prod"])
    cfg = tracking_from_args(args, tmp_path)
    assert (cfg.experiment_name, cfg.registered_model_name) == ("exp", "mlpp-prod")


def test_tracking_config_rejects_contradictions() -> None:
    with pytest.raises(ValueError, match="blank"):
        TrackingConfig(experiment_name=" ")
    with pytest.raises(ValueError, match="requires log_model"):
        TrackingConfig(log_model=False, registered_model_name="x")


# --- slow: real MLflow runs -------------------------------------------------------------


@pytest.mark.slow
def test_a_tracked_pipeline_run_records_params_metrics_and_artifacts(
    pipeline_cfg: PipelineConfig, tracking_cfg: TrackingConfig, frame: pd.DataFrame
) -> None:
    import mlflow

    from mlpp.pipeline import run_pipeline

    frame.to_csv(pipeline_cfg.data.train_dir / "train_02.csv", index=False)
    cfg = replace(
        pipeline_cfg,
        train=replace(pipeline_cfg.train, epochs=2, early_stopping_patience=10),
        tracking=replace(tracking_cfg, registered_model_name="mlpp-tests"),
    )
    result = run_pipeline(cfg, verbose=0)

    assert result.run_id is not None
    client = mlflow.MlflowClient(tracking_uri=tracking_cfg.tracking_uri)
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == result.session_dir.name

    params = run.data.params
    assert params["train.epochs"] == "2"
    assert params["data.n_features"] == "3"
    assert params["data.train_files"] == "train_01.csv,train_02.csv"
    assert run.data.tags["mlpp.schema_version"] == "2"

    # One step per epoch across both stages, on a single global axis.
    history = client.get_metric_history(result.run_id, "train_loss")
    assert [m.step for m in history] == [0, 1, 2, 3]
    assert {m.key for m in client.get_metric_history(result.run_id, "val_loss")} == {"val_loss"}
    assert "eval.stage_02.test_A.rmse" in run.data.metrics
    assert run.data.metrics["test_rmse_mean"] == pytest.approx(result.rows[-1].metrics.rmse)

    session_files = {Path(a.path).name for a in client.list_artifacts(result.run_id, "session")}
    assert {"manifest.json", "best_model.onnx", "best_model.pt"} <= session_files

    logged = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], output_format="list"
    )
    model = next(m for m in logged if m.source_run_id == result.run_id)
    served = mlflow.pyfunc.load_model(f"models:/{model.model_id}")
    batch = np.zeros((2, 3, 1), dtype=np.float32)
    assert np.asarray(next(iter(served.predict(batch).values()))).shape == (2, 1)
    assert client.get_registered_model("mlpp-tests").name == "mlpp-tests"


@pytest.mark.slow
def test_a_run_that_raises_is_closed_as_failed(
    pipeline_cfg: PipelineConfig, tracking_cfg: TrackingConfig, frame: pd.DataFrame
) -> None:
    import mlflow

    from mlpp.pipeline import run_pipeline

    poisoned = frame.copy()
    poisoned.loc[0, "target"] = np.nan  # the target is not imputed, so the loss goes NaN
    poisoned.to_csv(pipeline_cfg.data.train_dir / "train_01.csv", index=False)
    cfg = replace(
        pipeline_cfg,
        train=replace(pipeline_cfg.train, validation_split=0.0),
        tracking=tracking_cfg,
    )
    with pytest.raises(TrainingError):
        run_pipeline(cfg, verbose=0)

    client = mlflow.MlflowClient(tracking_uri=tracking_cfg.tracking_uri)
    experiment = client.get_experiment_by_name(tracking_cfg.experiment_name)
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"


@pytest.mark.slow
def test_an_unreachable_tracking_server_fails_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    cfg = TrackingConfig(enabled=True, tracking_uri="http://127.0.0.1:9")
    with pytest.raises(TrackingError, match="cannot start an MLflow run"):  # noqa: SIM117
        with start_tracking(cfg, run_name="never"):
            pytest.fail("the block must not run without a tracker")

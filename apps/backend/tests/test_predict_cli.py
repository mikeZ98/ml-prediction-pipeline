"""`mlpp-predict` argument handling and exit codes.

The train CLI has never had a test module; this covers the predict side from the
start. Every test runs a real session through the real scorer, so all are slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlpp.config import PipelineConfig
from mlpp.predict_cli import PREDICTION_COLUMN, main
from mlpp.session import MANIFEST_FILE

pytestmark = pytest.mark.slow


def _session_dir(pipeline_cfg: PipelineConfig) -> Path:
    from mlpp.pipeline import run_pipeline

    return run_pipeline(pipeline_cfg, verbose=0).session_dir


def _input_csv(frame: pd.DataFrame, tmp_path: Path, *, drop_target: bool = True) -> Path:
    payload = frame.drop(columns=["target"]) if drop_target else frame
    path = tmp_path / "to_score.csv"
    payload.to_csv(path, index=False)
    return path


def test_end_to_end_writes_predictions(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame, tmp_path: Path
) -> None:
    session = _session_dir(pipeline_cfg)
    source = _input_csv(frame, tmp_path)
    out = tmp_path / "preds.csv"

    code = main(
        ["--session", str(session), "--input", str(source), "--output", str(out), "--quiet"]
    )

    assert code == 0
    written = pd.read_csv(out)
    assert list(written.columns) == [PREDICTION_COLUMN]
    assert len(written) == len(frame)


def test_keep_inputs_writes_source_columns_alongside(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame, tmp_path: Path
) -> None:
    session = _session_dir(pipeline_cfg)
    source = _input_csv(frame, tmp_path)
    out = tmp_path / "preds.csv"

    main(
        [
            "--session",
            str(session),
            "--input",
            str(source),
            "--output",
            str(out),
            "--keep-inputs",
            "--quiet",
        ]
    )

    written = pd.read_csv(out)
    assert PREDICTION_COLUMN in written.columns
    assert "feature_01" in written.columns


def test_predictions_are_never_written_into_the_session(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame, tmp_path: Path
) -> None:
    """Session directories are reproducible training artifacts, not scratch space."""
    session = _session_dir(pipeline_cfg)
    before = {p.name for p in session.iterdir()}
    out = tmp_path / "preds.csv"

    main(
        [
            "--session",
            str(session),
            "--input",
            str(_input_csv(frame, tmp_path)),
            "--output",
            str(out),
            "--quiet",
        ]
    )

    assert {p.name for p in session.iterdir()} == before


def test_missing_session_directory_exits_non_zero(
    frame: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--session",
            str(tmp_path / "nope"),
            "--input",
            str(_input_csv(frame, tmp_path)),
            "--output",
            str(tmp_path / "preds.csv"),
        ]
    )

    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_schema_mismatched_input_exits_non_zero(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame, tmp_path: Path
) -> None:
    session = _session_dir(pipeline_cfg)
    short = frame.drop(columns=["target", "feature_01"])
    source = tmp_path / "short.csv"
    short.to_csv(source, index=False)

    code = main(
        [
            "--session",
            str(session),
            "--input",
            str(source),
            "--output",
            str(tmp_path / "preds.csv"),
        ]
    )
    assert code == 1


def test_lenient_schema_accepts_a_missing_column(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame, tmp_path: Path
) -> None:
    session = _session_dir(pipeline_cfg)
    short = frame.drop(columns=["target", "feature_01"])
    source = tmp_path / "short.csv"
    short.to_csv(source, index=False)
    out = tmp_path / "preds.csv"

    code = main(
        [
            "--session",
            str(session),
            "--input",
            str(source),
            "--output",
            str(out),
            "--lenient-schema",
            "--quiet",
        ]
    )
    assert code == 0
    assert len(pd.read_csv(out)) == len(short)


def test_version_mismatch_exits_non_zero(
    pipeline_cfg: PipelineConfig,
    frame: pd.DataFrame,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject-don't-migrate must reach the operator as a clean error, not a traceback."""
    session = _session_dir(pipeline_cfg)
    manifest_path = session / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    code = main(
        [
            "--session",
            str(session),
            "--input",
            str(_input_csv(frame, tmp_path)),
            "--output",
            str(tmp_path / "preds.csv"),
        ]
    )
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_unseen_levels_warn_but_still_exit_zero(
    pipeline_cfg: PipelineConfig,
    frame: pd.DataFrame,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stray category must not kill a batch — it warns and scores anyway."""
    from dataclasses import replace

    categorical = replace(pipeline_cfg.data.columns, categorical_columns=("comp_active",))
    cfg = replace(pipeline_cfg, data=replace(pipeline_cfg.data, columns=categorical))
    session = _session_dir(cfg)

    intruder = frame.drop(columns=["target"]).copy()
    intruder.loc[intruder.index[:2], "comp_active"] = 99
    source = tmp_path / "intruder.csv"
    intruder.to_csv(source, index=False)
    out = tmp_path / "preds.csv"

    code = main(
        ["--session", str(session), "--input", str(source), "--output", str(out), "--quiet"]
    )

    assert code == 0
    assert "not seen during training" in capsys.readouterr().err
    assert len(pd.read_csv(out)) == len(intruder)

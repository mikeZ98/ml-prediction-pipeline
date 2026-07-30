"""Scoring an existing session. Every test here imports Keras — all marked slow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlpp.config import ColumnConfig, PipelineConfig
from mlpp.errors import ArtifactError, SchemaError, SchemaVersionError
from mlpp.session import MANIFEST_FILE, load_session

pytestmark = pytest.mark.slow


def _trained_session(pipeline_cfg: PipelineConfig) -> tuple[Path, ColumnConfig]:
    """Run the real pipeline once and hand back its session directory."""
    from mlpp.pipeline import run_pipeline

    result = run_pipeline(pipeline_cfg, verbose=0)
    return result.session_dir, pipeline_cfg.data.columns


def test_score_frame_returns_one_prediction_per_row(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame
) -> None:
    from mlpp.predict import load_model, score_frame

    session_dir, columns = _trained_session(pipeline_cfg)
    session = load_session(session_dir, columns)
    result = score_frame(session, load_model(session), frame)

    assert result.predictions.shape == (len(frame),)
    assert np.isfinite(result.predictions).all()


def test_predictions_are_in_engineering_units(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame
) -> None:
    """Skipping `inverse_target` would leave predictions in scaled space.

    The target is standardised during fit, so scaled output clusters near zero with
    unit spread. Asserting the prediction mean sits near the *training* target mean
    rather than near zero is what catches a missing inverse transform.
    """
    from mlpp.predict import load_model, score_frame

    session_dir, columns = _trained_session(pipeline_cfg)
    session = load_session(session_dir, columns)
    result = score_frame(session, load_model(session), frame)

    target = frame[columns.output_column].to_numpy()
    offset = abs(result.predictions.mean() - target.mean())
    assert offset < 3 * target.std(), (
        f"prediction mean {result.predictions.mean():.3f} is far from target mean "
        f"{target.mean():.3f} — inverse_target may have been skipped"
    )


def test_target_column_is_not_required(pipeline_cfg: PipelineConfig, frame: pd.DataFrame) -> None:
    """Real inference input has no truth column — the `y=None` path must score."""
    from mlpp.predict import load_model, score_frame

    session_dir, columns = _trained_session(pipeline_cfg)
    session = load_session(session_dir, columns)
    unlabelled = frame.drop(columns=[columns.output_column])

    result = score_frame(session, load_model(session), unlabelled)
    assert result.predictions.shape == (len(unlabelled),)


def test_missing_input_column_raises_schema_error(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame
) -> None:
    from mlpp.predict import load_model, score_frame

    session_dir, columns = _trained_session(pipeline_cfg)
    session = load_session(session_dir, columns)
    with pytest.raises(SchemaError, match="feature_01"):
        score_frame(session, load_model(session), frame.drop(columns=["feature_01"]))


def test_unseen_categorical_levels_are_reported(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame
) -> None:
    """The blind spot behind `handle_unknown="ignore"`, surfaced instead of swallowed."""
    from mlpp.predict import load_model, score_frame

    categorical = replace(pipeline_cfg.data.columns, categorical_columns=("comp_active",))
    cfg = replace(pipeline_cfg, data=replace(pipeline_cfg.data, columns=categorical))
    session_dir, columns = _trained_session(cfg)
    session = load_session(session_dir, columns)

    intruder = frame.copy()
    intruder.loc[intruder.index[:2], "comp_active"] = 99

    result = score_frame(session, load_model(session), intruder)
    assert result.has_unseen
    assert 99 in result.unseen["comp_active"]
    assert result.unseen_rows == 2
    assert "comp_active" in result.describe_unseen()


def test_clean_input_reports_nothing_unseen(
    pipeline_cfg: PipelineConfig, frame: pd.DataFrame
) -> None:
    from mlpp.predict import load_model, score_frame

    session_dir, columns = _trained_session(pipeline_cfg)
    session = load_session(session_dir, columns)
    result = score_frame(session, load_model(session), frame)

    assert not result.has_unseen
    assert result.unseen_rows == 0
    assert result.describe_unseen() == ""


def test_load_model_rejects_a_session_with_no_model(
    pipeline_cfg: PipelineConfig, tmp_path: Path
) -> None:
    """A manifest recording no model role must fail loudly, not return None."""
    from mlpp.predict import load_model

    session_dir, columns = _trained_session(pipeline_cfg)
    manifest_path = session_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [a for a in manifest["artifacts"] if a["role"] != "best_model"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    session = load_session(session_dir, columns)
    with pytest.raises(ArtifactError, match="no 'best_model' artifact"):
        load_model(session)


def test_scoring_a_version_mismatched_session_is_rejected(
    pipeline_cfg: PipelineConfig,
) -> None:
    """The reject-don't-migrate rule must hold through the predict path too."""
    session_dir, columns = _trained_session(pipeline_cfg)
    manifest_path = session_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = manifest["schema_version"] + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SchemaVersionError):
        load_session(session_dir, columns)


def test_committed_reference_run_is_scorable(frame: pd.DataFrame) -> None:
    """Guards the committed example against the silent rot s-03 fixed."""
    from mlpp.predict import load_model, score_frame

    example = Path(__file__).resolve().parents[3] / "OUTPUTS" / "example"
    if not (example / MANIFEST_FILE).is_file():
        pytest.skip("no committed reference run to score")

    session = load_session(example, ColumnConfig())
    columns = session.manifest.features
    payload = pd.DataFrame(
        {
            name: frame.get(name, pd.Series(0.0, index=frame.index))
            for name in columns.numeric_columns
        }
    )
    result = score_frame(session, load_model(session), payload)
    assert result.predictions.shape == (len(payload),)

"""Contract tests for the session manifest. TF-free — part of the fast suite."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mlpp.config import DataConfig
from mlpp.errors import ArtifactError, SchemaVersionError
from mlpp.preprocess import Preprocessor
from mlpp.session import (
    ENCODERS_FILE,
    MANIFEST_FILE,
    OUTPUT_SCALER_FILE,
    ROLE_ENCODERS,
    ROLE_OUTPUT_SCALER,
    ROLE_SCALER,
    ROLE_STAGE_MODEL,
    SCALER_FILE,
    SCHEMA_VERSION,
    FeatureContract,
    SessionWriter,
    load_session,
    prediction_analysis_filename,
    read_fitted_state,
    read_manifest,
    stage_history_filename,
    stage_model_filename,
    training_curves_filename,
    write_fitted_state,
)

CONTRACT = FeatureContract(
    numeric_columns=("feature_01", "feature_02"),
    categorical_columns=("comp_active",),
    output_column="target",
    feature_names=("feature_01", "feature_02", "comp_active_0", "comp_active_1"),
)


def _written_session(tmp_path: Path) -> SessionWriter:
    writer = SessionWriter(tmp_path / "session")
    writer.set_features(CONTRACT)
    writer.register(ROLE_ENCODERS, "encoders.gz")
    writer.flush()
    return writer


# --- directory lifecycle ---------------------------------------------------


def test_writer_creates_its_directory(tmp_path: Path) -> None:
    writer = SessionWriter(tmp_path / "nested" / "session")
    assert writer.session_dir.is_dir()


def test_create_uses_timestamp(tmp_path: Path) -> None:
    writer = SessionWriter.create(tmp_path, now=datetime(2026, 7, 29, 8, 30, 15))
    assert writer.session_dir.name == "2026-07-29_08-30-15"


def test_injected_clock_governs_directory_name_and_created_stamp(tmp_path: Path) -> None:
    """An injected clock that only names the directory is worse than none — the
    manifest would claim a different moment than the directory it sits in."""
    moment = datetime(2026, 7, 29, 8, 30, 15)
    writer = SessionWriter.create(tmp_path, now=moment)
    writer.set_features(CONTRACT)
    writer.flush()
    assert read_manifest(writer.session_dir).created == moment.isoformat(timespec="seconds")


def test_directory_exists_before_features_are_known(tmp_path: Path) -> None:
    """The dir is needed before training; the contract only exists after fit."""
    writer = SessionWriter(tmp_path / "session")
    assert writer.session_dir.is_dir()
    assert not writer.has_features


# --- filename ownership ----------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "stage_or_tag", "expected"),
    [
        (stage_model_filename, 1, "model_iter_01.keras"),
        (stage_model_filename, 12, "model_iter_12.keras"),
        (stage_history_filename, 2, "history_train_02.csv"),
        (training_curves_filename, "train_01", "training_curves_train_01.png"),
        (prediction_analysis_filename, "tr01_te02", "prediction_analysis_tr01_te02.html"),
    ],
)
def test_filename_builders(builder, stage_or_tag, expected) -> None:
    assert builder(stage_or_tag) == expected


def test_register_returns_a_path_inside_the_session(tmp_path: Path) -> None:
    writer = SessionWriter(tmp_path / "session")
    path = writer.register(ROLE_STAGE_MODEL, stage_model_filename(1))
    assert path == writer.session_dir / "model_iter_01.keras"


# --- manifest construction -------------------------------------------------


def test_inventory_accumulates_across_stages(tmp_path: Path) -> None:
    """The obvious 'write once at fit' ordering would record only stage 1."""
    writer = SessionWriter(tmp_path / "session")
    writer.set_features(CONTRACT)
    for stage in (1, 2, 3):
        writer.register(ROLE_STAGE_MODEL, stage_model_filename(stage))
    writer.flush()

    manifest = read_manifest(writer.session_dir)
    assert manifest.filenames_for(ROLE_STAGE_MODEL) == (
        "model_iter_01.keras",
        "model_iter_02.keras",
        "model_iter_03.keras",
    )


def test_registering_the_same_artifact_twice_is_a_noop(tmp_path: Path) -> None:
    writer = SessionWriter(tmp_path / "session")
    writer.set_features(CONTRACT)
    writer.register(ROLE_ENCODERS, "encoders.gz")
    writer.register(ROLE_ENCODERS, "encoders.gz")
    assert len(writer.build_manifest().artifacts) == 1


def test_set_features_is_idempotent(tmp_path: Path) -> None:
    writer = SessionWriter(tmp_path / "session")
    writer.set_features(CONTRACT)
    writer.set_features(CONTRACT)
    assert writer.has_features


def test_conflicting_feature_contract_is_rejected(tmp_path: Path) -> None:
    """A session is fitted once; a second, different contract signals a bug."""
    writer = SessionWriter(tmp_path / "session")
    writer.set_features(CONTRACT)
    other = FeatureContract((), (), "target", ("x",))
    with pytest.raises(ArtifactError, match="fitted once"):
        writer.set_features(other)


def test_flush_before_features_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="before the feature contract is set"):
        SessionWriter(tmp_path / "session").flush()


# --- round trip ------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    writer = _written_session(tmp_path)
    manifest = read_manifest(writer.session_dir)
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.features == CONTRACT
    assert manifest.artifacts == writer.build_manifest().artifacts


def test_manifest_is_human_readable(tmp_path: Path) -> None:
    writer = _written_session(tmp_path)
    text = (writer.session_dir / MANIFEST_FILE).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n  " in text, "manifest should be indented, not minified"
    assert json.loads(text)["features"]["output_column"] == "target"


def test_flush_is_repeatable(tmp_path: Path) -> None:
    writer = _written_session(tmp_path)
    writer.register(ROLE_STAGE_MODEL, stage_model_filename(2))
    writer.flush()
    assert read_manifest(writer.session_dir).filenames_for(ROLE_STAGE_MODEL) == (
        "model_iter_02.keras",
    )


# --- rejection cases -------------------------------------------------------


def test_absent_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not a session directory"):
        read_manifest(tmp_path / "nope")


def test_empty_directory_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArtifactError, match=f"no {MANIFEST_FILE}"):
        read_manifest(empty)


def test_pre_manifest_session_names_the_version(tmp_path: Path) -> None:
    """The committed OUTPUTS/example/ is exactly this case."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "feature_config.json").write_text('{"input_columns": []}', encoding="utf-8")
    with pytest.raises(SchemaVersionError, match="predates the manifest contract") as exc:
        read_manifest(legacy)
    assert str(SCHEMA_VERSION) in str(exc.value)
    assert "mlpp-train" in str(exc.value), "error must say how to fix it"


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        read_manifest(session)


def test_non_object_manifest_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / MANIFEST_FILE).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ArtifactError, match="must contain a JSON object"):
        read_manifest(session)


def test_missing_version_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / MANIFEST_FILE).write_text(json.dumps({"features": {}}), encoding="utf-8")
    with pytest.raises(SchemaVersionError, match="no 'schema_version'"):
        read_manifest(session)


def test_future_version_names_both_versions(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    future = SCHEMA_VERSION + 1
    (session / MANIFEST_FILE).write_text(
        json.dumps({"schema_version": future, "features": CONTRACT.as_dict()}), encoding="utf-8"
    )
    with pytest.raises(SchemaVersionError) as exc:
        read_manifest(session)
    message = str(exc.value)
    assert str(future) in message and str(SCHEMA_VERSION) in message


def test_non_integer_version_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / MANIFEST_FILE).write_text(
        json.dumps({"schema_version": "1", "features": {}}), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="must be an integer"):
        read_manifest(session)


def test_missing_features_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / MANIFEST_FILE).write_text(
        json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="missing 'features'"):
        read_manifest(session)


def test_incomplete_feature_contract_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / MANIFEST_FILE).write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "features": {"output_column": "t"}}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="missing keys"):
        read_manifest(session)


def test_malformed_artifact_entry_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / MANIFEST_FILE).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "features": CONTRACT.as_dict(),
                "artifacts": [{"role": "encoders"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="needs 'role' and 'filename'"):
        read_manifest(session)


def test_schema_version_error_is_an_artifact_error() -> None:
    """Callers that only care about 'bad artifact' should not need the subclass."""
    assert issubclass(SchemaVersionError, ArtifactError)


# --- fitted-state persistence ----------------------------------------------


def test_writer_reopens_an_existing_directory(tmp_path: Path) -> None:
    """Constructing over an existing session dir must not fail or clobber it."""
    session = tmp_path / "session"
    session.mkdir()
    (session / "keepme.txt").write_text("x", encoding="utf-8")
    SessionWriter(session)
    assert (session / "keepme.txt").is_file()


def test_fitted_state_round_trip(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    """Persisted estimators must reproduce the original transform exactly."""
    pre = Preprocessor(data_cfg).fit(frame)
    writer = SessionWriter(data_cfg.output_dir / "session")
    writer.set_features(
        FeatureContract(
            pre.schema.numeric, pre.schema.categorical, pre.schema.output, pre.feature_names
        )
    )
    write_fitted_state(writer, pre.fitted_state)
    writer.flush()

    restored = Preprocessor.restore(
        data_cfg, read_fitted_state(writer.session_dir), pre.schema, pre.feature_names
    )
    np.testing.assert_allclose(restored.transform(frame)[0], pre.transform(frame)[0])


def test_write_fitted_state_registers_all_three_files(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    pre = Preprocessor(data_cfg).fit(frame)
    writer = SessionWriter(data_cfg.output_dir / "session")
    write_fitted_state(writer, pre.fitted_state)
    roles = {a.role for a in writer._artifacts}  # noqa: SLF001 - inventory is the assertion
    assert roles == {ROLE_SCALER, ROLE_OUTPUT_SCALER, ROLE_ENCODERS}
    for name in (SCALER_FILE, OUTPUT_SCALER_FILE, ENCODERS_FILE):
        assert (writer.session_dir / name).is_file()


def test_encoders_file_holds_only_the_encoder(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    """Previously this bundled schema + feature_names — one of the three copies."""
    pre = Preprocessor(data_cfg).fit(frame)
    writer = SessionWriter(data_cfg.output_dir / "session")
    write_fitted_state(writer, pre.fitted_state)
    blob = joblib.load(writer.session_dir / ENCODERS_FILE)
    assert not isinstance(blob, dict), "encoders.gz must be the encoder alone, not a bundle"


def test_read_fitted_state_reports_every_missing_file(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    with pytest.raises(ArtifactError, match="missing fitted-state artifacts") as exc:
        read_fitted_state(session)
    for name in (SCALER_FILE, OUTPUT_SCALER_FILE, ENCODERS_FILE):
        assert name in str(exc.value)


def test_read_fitted_state_reports_a_partial_session(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    pre = Preprocessor(data_cfg).fit(frame)
    writer = SessionWriter(data_cfg.output_dir / "session")
    write_fitted_state(writer, pre.fitted_state)
    (writer.session_dir / ENCODERS_FILE).unlink()
    with pytest.raises(ArtifactError, match=ENCODERS_FILE):
        read_fitted_state(writer.session_dir)


# --- load_session round trip -----------------------------------------------


def _write_session(frame: pd.DataFrame, data_cfg: DataConfig) -> tuple[SessionWriter, Preprocessor]:
    pre = Preprocessor(data_cfg).fit(frame)
    writer = SessionWriter(data_cfg.output_dir / "session")
    writer.set_features(
        FeatureContract(
            pre.schema.numeric, pre.schema.categorical, pre.schema.output, pre.feature_names
        )
    )
    write_fitted_state(writer, pre.fitted_state)
    writer.flush()
    return writer, pre


def test_load_session_round_trip_is_numerically_identical(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    """The assertion whose absence let the committed reference run rot unnoticed."""
    writer, pre = _write_session(frame, data_cfg)
    loaded = load_session(writer.session_dir, data_cfg)

    x_before, y_before = pre.transform(frame)
    x_after, y_after = loaded.preprocessor.transform(frame)
    np.testing.assert_allclose(x_after, x_before)
    assert y_before is not None and y_after is not None
    np.testing.assert_allclose(y_after, y_before)


def test_load_session_restores_the_feature_contract(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    writer, pre = _write_session(frame, data_cfg)
    loaded = load_session(writer.session_dir, data_cfg)
    assert loaded.preprocessor.feature_names == pre.feature_names
    assert loaded.preprocessor.n_features == pre.n_features
    assert loaded.preprocessor.schema == pre.schema
    assert loaded.session_dir == writer.session_dir


def test_load_session_inverse_target_round_trips(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    writer, _ = _write_session(frame, data_cfg)
    loaded = load_session(writer.session_dir, data_cfg)
    _, y = loaded.preprocessor.transform(frame)
    assert y is not None
    np.testing.assert_allclose(
        loaded.preprocessor.inverse_target(y), frame["target"].to_numpy(), atol=1e-6
    )


def test_load_session_rejects_a_listed_but_missing_file(
    frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    """A manifest that lies about its contents must fail, not degrade silently."""
    writer, _ = _write_session(frame, data_cfg)
    (writer.session_dir / SCALER_FILE).unlink()
    with pytest.raises(ArtifactError, match="not on disk") as exc:
        load_session(writer.session_dir, data_cfg)
    assert SCALER_FILE in str(exc.value)


def test_load_session_rejects_a_pre_manifest_directory(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "feature_config.json").write_text('{"input_columns": []}', encoding="utf-8")
    with pytest.raises(SchemaVersionError, match="predates the manifest contract"):
        load_session(legacy, DataConfig(tmp_path, tmp_path, tmp_path))


def test_load_session_does_not_import_tensorflow(frame: pd.DataFrame, data_cfg: DataConfig) -> None:
    """Reading a session must never require the training stack.

    Asserted in a subprocess on purpose. `sys.modules` is process-global, so an
    in-process check would pass or fail depending on whether a Keras test ran
    first — it would measure suite ordering, not this module's imports.
    """
    writer, _ = _write_session(frame, data_cfg)
    program = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from mlpp.config import DataConfig
        from mlpp.session import load_session

        cfg = DataConfig(
            train_dir=Path({str(data_cfg.train_dir)!r}),
            test_dir=Path({str(data_cfg.test_dir)!r}),
            output_dir=Path({str(data_cfg.output_dir)!r}),
            input_columns={data_cfg.input_columns!r},
            output_column={data_cfg.output_column!r},
        )
        loaded = load_session(Path({str(writer.session_dir)!r}), cfg)
        assert loaded.preprocessor.n_features > 0
        for banned in ("tensorflow", "keras"):
            assert banned not in sys.modules, banned + " was imported"
    """)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

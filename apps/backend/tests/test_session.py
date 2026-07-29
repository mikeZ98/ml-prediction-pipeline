"""Contract tests for the session manifest. TF-free — part of the fast suite."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from mlpp.errors import ArtifactError, SchemaVersionError
from mlpp.session import (
    MANIFEST_FILE,
    ROLE_ENCODERS,
    ROLE_STAGE_MODEL,
    SCHEMA_VERSION,
    FeatureContract,
    SessionWriter,
    prediction_analysis_filename,
    read_manifest,
    stage_history_filename,
    stage_model_filename,
    training_curves_filename,
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

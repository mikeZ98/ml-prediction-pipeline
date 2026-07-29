"""Session-directory contract: the single owner of every artifact filename.

Before this module, four modules named files in a session directory and three of
them persisted overlapping copies of the feature contract with no cross-check —
which is how the committed reference run drifted out of compatibility unnoticed.
Everything written into a session directory now goes through `SessionWriter`, and
`manifest.json` is the single source of truth for what that directory contains.

TensorFlow-free by design: inspecting or validating a session must never require
importing Keras.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import joblib

from mlpp.config import DataConfig
from mlpp.errors import ArtifactError, SchemaVersionError
from mlpp.preprocess import FeatureSchema, FittedState, Preprocessor

#: Bumped whenever the manifest layout changes incompatibly. The loader rejects
#: anything it does not recognise rather than guessing.
SCHEMA_VERSION: Final = 1

MANIFEST_FILE: Final = "manifest.json"
SESSION_FORMAT: Final = "%Y-%m-%d_%H-%M-%S"

# Every filename that may appear in a session directory. Nothing outside this
# module is allowed to spell these out.
SCALER_FILE: Final = "scaler.gz"
OUTPUT_SCALER_FILE: Final = "output_scaler.gz"
ENCODERS_FILE: Final = "encoders.gz"
METRICS_FILE: Final = "test_metrics.csv"
BEST_MODEL_FILE: Final = "best_model.keras"
TRAINING_LOG_FILE: Final = "training_log.csv"

# Artifact roles recorded in the manifest inventory.
ROLE_SCALER: Final = "scaler"
ROLE_OUTPUT_SCALER: Final = "output_scaler"
ROLE_ENCODERS: Final = "encoders"
ROLE_METRICS: Final = "metrics"
ROLE_BEST_MODEL: Final = "best_model"
ROLE_TRAINING_LOG: Final = "training_log"
ROLE_STAGE_MODEL: Final = "stage_model"
ROLE_STAGE_HISTORY: Final = "stage_history"
ROLE_TRAINING_CURVES: Final = "training_curves"
ROLE_PREDICTION_ANALYSIS: Final = "prediction_analysis"

#: Filenames written by the pre-manifest layout. Used only to tell "this is an old
#: session" apart from "this is not a session at all" when producing errors.
_LEGACY_MARKERS: Final = ("feature_config.json", ENCODERS_FILE, SCALER_FILE, BEST_MODEL_FILE)


def stage_model_filename(stage: int) -> str:
    """Model snapshot for training stage `stage`."""
    return f"model_iter_{stage:02d}.keras"


def stage_history_filename(stage: int) -> str:
    """Per-epoch history for training stage `stage`."""
    return f"history_train_{stage:02d}.csv"


def training_curves_filename(tag: str) -> str:
    """Loss/MAE curve image for `tag`."""
    return f"training_curves_{tag}.png"


def prediction_analysis_filename(tag: str) -> str:
    """Interactive prediction report for `tag`."""
    return f"prediction_analysis_{tag}.html"


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """The column contract a trained model expects, recorded exactly once."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    output_column: str
    #: Post-one-hot names, in the order they occupy axis 1 of X.
    feature_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "output_column": self.output_column,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: Any, source: Path) -> FeatureContract:
        if not isinstance(payload, dict):
            raise ArtifactError(f"{source}: 'features' must be an object")
        missing = {
            "numeric_columns",
            "categorical_columns",
            "output_column",
            "feature_names",
        } - payload.keys()
        if missing:
            raise ArtifactError(f"{source}: 'features' is missing keys: {sorted(missing)}")
        return cls(
            numeric_columns=tuple(payload["numeric_columns"]),
            categorical_columns=tuple(payload["categorical_columns"]),
            output_column=str(payload["output_column"]),
            feature_names=tuple(payload["feature_names"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """One file in the session directory, tagged with what it is."""

    role: str
    filename: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "filename": self.filename}


@dataclass(frozen=True, slots=True)
class SessionManifest:
    """What a session directory contains and which contract it was written under."""

    schema_version: int
    created: str
    features: FeatureContract
    artifacts: tuple[ArtifactEntry, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created": self.created,
            "features": self.features.as_dict(),
            "artifacts": [a.as_dict() for a in self.artifacts],
        }

    def filenames_for(self, role: str) -> tuple[str, ...]:
        """Every filename recorded under `role`, in registration order."""
        return tuple(a.filename for a in self.artifacts if a.role == role)


class SessionWriter:
    """Owns one session directory: creates it, names its files, writes its manifest.

    Lifecycle matters here. The directory must exist before training starts, but
    the feature contract does not exist until the preprocessor is fitted, and the
    artifact inventory keeps growing as training stages complete. So: construct
    early, `set_features` at first fit, `register` as files are written, `flush`
    once at the end.
    """

    def __init__(self, session_dir: Path, created: datetime | None = None) -> None:
        self._dir = session_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._features: FeatureContract | None = None
        self._artifacts: list[ArtifactEntry] = []
        self._created = (created or datetime.now()).isoformat(timespec="seconds")

    @classmethod
    def create(cls, base: Path, now: datetime | None = None) -> SessionWriter:
        """Open a new timestamped session directory under `base`.

        `now` is injectable so tests do not depend on the wall clock; it governs
        both the directory name and the manifest's `created` stamp, so the two
        can never disagree.
        """
        moment = now or datetime.now()
        return cls(base / moment.strftime(SESSION_FORMAT), created=moment)

    @property
    def session_dir(self) -> Path:
        return self._dir

    @property
    def has_features(self) -> bool:
        return self._features is not None

    def set_features(self, contract: FeatureContract) -> None:
        """Record the feature contract. Idempotent; conflicting rewrites are a bug."""
        if self._features is not None and self._features != contract:
            raise ArtifactError(
                "refusing to overwrite the session feature contract with a different one — "
                "a session is fitted once"
            )
        self._features = contract

    def register(self, role: str, filename: str) -> Path:
        """Record `filename` under `role` and return the full path to write to.

        Registering the same (role, filename) twice is a no-op, so a caller that
        rewrites a file across stages does not bloat the inventory.
        """
        entry = ArtifactEntry(role=role, filename=filename)
        if entry not in self._artifacts:
            self._artifacts.append(entry)
        return self._dir / filename

    def path_for(self, filename: str) -> Path:
        """Full path for a filename without registering it (for reads/probes)."""
        return self._dir / filename

    def build_manifest(self) -> SessionManifest:
        if self._features is None:
            raise ArtifactError("cannot build a manifest before the feature contract is set")
        return SessionManifest(
            schema_version=SCHEMA_VERSION,
            created=self._created,
            features=self._features,
            artifacts=tuple(self._artifacts),
        )

    def flush(self) -> Path:
        """Write `manifest.json`. Safe to call repeatedly; last call wins."""
        manifest = self.build_manifest()
        path = self._dir / MANIFEST_FILE
        path.write_text(json.dumps(manifest.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def write_fitted_state(writer: SessionWriter, state: FittedState) -> None:
    """Persist the fitted estimators and record them in the inventory.

    Only the estimators are pickled. The schema and feature names go in the
    manifest — keeping them out of `encoders.gz` is what stops the contract from
    having two homes that can disagree.
    """
    joblib.dump(state.scaler, writer.register(ROLE_SCALER, SCALER_FILE))
    joblib.dump(state.output_scaler, writer.register(ROLE_OUTPUT_SCALER, OUTPUT_SCALER_FILE))
    joblib.dump(state.onehot, writer.register(ROLE_ENCODERS, ENCODERS_FILE))


def read_fitted_state(session_dir: Path) -> FittedState:
    """Load the fitted estimators, failing clearly when any file is absent."""
    paths = {
        SCALER_FILE: session_dir / SCALER_FILE,
        OUTPUT_SCALER_FILE: session_dir / OUTPUT_SCALER_FILE,
        ENCODERS_FILE: session_dir / ENCODERS_FILE,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ArtifactError(f"missing fitted-state artifacts in {session_dir}: {sorted(missing)}")
    return FittedState(
        scaler=joblib.load(paths[SCALER_FILE]),
        output_scaler=joblib.load(paths[OUTPUT_SCALER_FILE]),
        onehot=joblib.load(paths[ENCODERS_FILE]),
    )


def read_manifest(session_dir: Path) -> SessionManifest:
    """Load and validate the manifest for `session_dir`.

    Raises `SchemaVersionError` when the session predates this contract or carries
    a version this build does not support, and `ArtifactError` when the manifest is
    absent, unparseable, or structurally wrong.
    """
    path = session_dir / MANIFEST_FILE
    if not path.is_file():
        raise _missing_manifest_error(session_dir, path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path} must contain a JSON object, got {type(payload).__name__}")

    version = payload.get("schema_version")
    if version is None:
        raise SchemaVersionError(
            f"{path} has no 'schema_version'; this build writes and reads version "
            f"{SCHEMA_VERSION}. Regenerate the session with `mlpp-train`."
        )
    if not isinstance(version, int) or isinstance(version, bool):
        raise ArtifactError(f"{path}: 'schema_version' must be an integer, got {version!r}")
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{path} was written under schema version {version}, but this build supports "
            f"version {SCHEMA_VERSION}. Artifacts are not migrated — regenerate the session "
            f"with `mlpp-train`."
        )

    if "features" not in payload:
        raise ArtifactError(f"{path} is missing 'features'")
    return SessionManifest(
        schema_version=version,
        created=str(payload.get("created", "")),
        features=FeatureContract.from_dict(payload["features"], path),
        artifacts=_entries_from(payload.get("artifacts", []), path),
    )


@dataclass(frozen=True, slots=True)
class LoadedSession:
    """A session restored from disk: what it claims, and a preprocessor to use."""

    session_dir: Path
    manifest: SessionManifest
    preprocessor: Preprocessor


def load_session(session_dir: Path, cfg: DataConfig, *, use_onehot: bool = True) -> LoadedSession:
    """Restore a session's manifest and a working Preprocessor from `session_dir`.

    Validates that every file the manifest lists is actually present — the
    cross-check whose absence let the committed reference run rot unnoticed.

    Deliberately does not load the Keras model: that would drag TensorFlow into
    this module, and scoring belongs to the predict CLI, not to the contract.

    Raises `SchemaVersionError` for a version this build cannot read, and
    `ArtifactError` when the manifest and the directory disagree.
    """
    manifest = read_manifest(session_dir)
    absent = [e.filename for e in manifest.artifacts if not (session_dir / e.filename).is_file()]
    if absent:
        raise ArtifactError(
            f"{session_dir}: manifest lists {len(absent)} file(s) that are not on disk: "
            f"{sorted(absent)}"
        )

    features = manifest.features
    schema = FeatureSchema(
        numeric=features.numeric_columns,
        categorical=features.categorical_columns,
        output=features.output_column,
    )
    preprocessor = Preprocessor.restore(
        cfg,
        read_fitted_state(session_dir),
        schema,
        features.feature_names,
        use_onehot=use_onehot,
    )
    return LoadedSession(session_dir=session_dir, manifest=manifest, preprocessor=preprocessor)


def _missing_manifest_error(session_dir: Path, path: Path) -> ArtifactError:
    """Tell 'old session' apart from 'not a session', so the message is actionable."""
    if not session_dir.is_dir():
        return ArtifactError(f"not a session directory: {session_dir}")
    if any((session_dir / marker).exists() for marker in _LEGACY_MARKERS):
        return SchemaVersionError(
            f"{session_dir} predates the manifest contract (no {MANIFEST_FILE}); this build "
            f"supports version {SCHEMA_VERSION}. Artifacts are not migrated — regenerate the "
            f"session with `mlpp-train`."
        )
    return ArtifactError(f"no {MANIFEST_FILE} in {session_dir}")


def _entries_from(raw: Any, source: Path) -> tuple[ArtifactEntry, ...]:
    if not isinstance(raw, list):
        raise ArtifactError(f"{source}: 'artifacts' must be a list")
    entries: list[ArtifactEntry] = []
    for item in raw:
        if not isinstance(item, dict) or "role" not in item or "filename" not in item:
            raise ArtifactError(
                f"{source}: each artifact needs 'role' and 'filename', got {item!r}"
            )
        entries.append(ArtifactEntry(role=str(item["role"]), filename=str(item["filename"])))
    return tuple(entries)

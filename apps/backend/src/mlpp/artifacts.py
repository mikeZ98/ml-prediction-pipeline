"""Run directories and JSON side-car artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mlpp.errors import ArtifactError
from mlpp.preprocess import FeatureSchema

FEATURE_CONFIG_FILE = "feature_config.json"
SESSION_FORMAT = "%Y-%m-%d_%H-%M-%S"


def make_session_dir(base: Path, now: datetime | None = None) -> Path:
    """Create and return a timestamped run directory under `base`.

    `now` is injectable so tests do not depend on the wall clock.
    """
    base.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime(SESSION_FORMAT)
    session = base / stamp
    session.mkdir(parents=True, exist_ok=True)
    return session


def save_feature_config(
    schema: FeatureSchema, feature_names: tuple[str, ...], out_dir: Path
) -> Path:
    """Persist the exact column contract the trained model expects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FEATURE_CONFIG_FILE
    path.write_text(
        json.dumps(
            {
                "numeric_columns": list(schema.numeric),
                "categorical_columns": list(schema.categorical),
                "output_column": schema.output,
                "feature_names": list(feature_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_feature_config(out_dir: Path) -> dict[str, object]:
    """Read back a feature config, raising ArtifactError if absent or malformed."""
    path = out_dir / FEATURE_CONFIG_FILE
    if not path.is_file():
        raise ArtifactError(f"missing {FEATURE_CONFIG_FILE} in {out_dir}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path} must contain a JSON object, got {type(payload).__name__}")
    missing = {"numeric_columns", "output_column", "feature_names"} - payload.keys()
    if missing:
        raise ArtifactError(f"{path} is missing keys: {sorted(missing)}")
    return dict(payload)

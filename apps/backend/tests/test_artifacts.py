from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from mlpp.artifacts import (
    FEATURE_CONFIG_FILE,
    load_feature_config,
    make_session_dir,
    save_feature_config,
)
from mlpp.config import DataConfig
from mlpp.errors import ArtifactError
from mlpp.preprocess import Preprocessor


def test_make_session_dir_uses_timestamp(tmp_path: Path) -> None:
    session = make_session_dir(tmp_path, now=datetime(2026, 7, 29, 8, 30, 15))
    assert session.name == "2026-07-29_08-30-15"
    assert session.is_dir()


def test_make_session_dir_creates_missing_base(tmp_path: Path) -> None:
    assert make_session_dir(tmp_path / "OUTPUTS").is_dir()


def test_make_session_dir_is_idempotent(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1)
    assert make_session_dir(tmp_path, now=now) == make_session_dir(tmp_path, now=now)


def test_feature_config_round_trip(
    frame: pd.DataFrame, data_cfg: DataConfig, tmp_path: Path
) -> None:
    pre = Preprocessor(data_cfg).fit(frame)
    save_feature_config(pre.schema, pre.feature_names, tmp_path)
    payload = load_feature_config(tmp_path)
    assert payload["output_column"] == "target"
    assert payload["feature_names"] == list(pre.feature_names)


def test_load_feature_config_reports_absence(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match=f"missing {FEATURE_CONFIG_FILE}"):
        load_feature_config(tmp_path)


def test_load_feature_config_rejects_malformed_json(tmp_path: Path) -> None:
    (tmp_path / FEATURE_CONFIG_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        load_feature_config(tmp_path)


def test_load_feature_config_rejects_incomplete_payload(tmp_path: Path) -> None:
    (tmp_path / FEATURE_CONFIG_FILE).write_text(
        json.dumps({"output_column": "t"}), encoding="utf-8"
    )
    with pytest.raises(ArtifactError, match="missing keys"):
        load_feature_config(tmp_path)

"""Dashboard logic that is testable without a browser.

Rendering is left to manual verification; what is pinned here is session discovery
and the error surface — the behaviour that decides whether an author sees a message
naming the problem or a traceback.

`list_sessions` is imported from `mlpp.dashboard.loaders`, which imports Streamlit
but not Keras, so these stay in the fast suite. Tests that genuinely load a model are
marked `slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlpp.config import ColumnConfig, DataConfig
from mlpp.dashboard.loaders import default_outputs_root, list_sessions
from mlpp.errors import ArtifactError, SchemaVersionError
from mlpp.session import MANIFEST_FILE, SCHEMA_VERSION, load_session

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SESSION = REPO_ROOT / "OUTPUTS" / "example"
APP_SCRIPT = REPO_ROOT / "apps" / "backend" / "src" / "mlpp" / "dashboard" / "app.py"


def test_default_outputs_root_resolves_to_the_repository_outputs() -> None:
    """Pins the `parents[N]` depth.

    Getting this wrong is silent: the app renders an empty "no sessions found" page
    instead of failing, which is exactly how it shipped wrong the first time.
    """
    assert default_outputs_root() == REPO_ROOT / "OUTPUTS"


def test_list_sessions_returns_newest_first(tmp_path: Path) -> None:
    """Session directories are timestamped, so name-descending is chronological."""
    for name in ("2026-07-30_10-46-45", "2026-07-30_13-08-29", "2026-07-30_12-38-54"):
        (tmp_path / name).mkdir()

    assert [p.name for p in list_sessions(tmp_path)] == [
        "2026-07-30_13-08-29",
        "2026-07-30_12-38-54",
        "2026-07-30_10-46-45",
    ]


def test_list_sessions_ignores_files(tmp_path: Path) -> None:
    (tmp_path / "a_session").mkdir()
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in list_sessions(tmp_path)] == ["a_session"]


def test_list_sessions_on_an_empty_directory(tmp_path: Path) -> None:
    assert list_sessions(tmp_path) == []


def test_list_sessions_on_an_absent_root(tmp_path: Path) -> None:
    """An empty or missing OUTPUTS/ is a starting state, not an error."""
    assert list_sessions(tmp_path / "does_not_exist") == []


def test_list_sessions_keeps_invalid_directories(tmp_path: Path) -> None:
    """Candidacy is existence, not validity — hiding a broken session hides the problem."""
    (tmp_path / "empty_dir").mkdir()
    assert [p.name for p in list_sessions(tmp_path)] == ["empty_dir"]


def test_a_directory_without_a_manifest_surfaces_artifact_error(tmp_path: Path) -> None:
    """The error the dashboard renders must name the missing file."""
    session = tmp_path / "no_manifest"
    session.mkdir()
    with pytest.raises(ArtifactError) as excinfo:
        load_session(session, ColumnConfig())
    assert MANIFEST_FILE in str(excinfo.value)


def test_a_version_mismatch_surfaces_schema_version_error(
    tmp_path: Path, frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    session = _minimal_session(tmp_path, frame, data_cfg, schema_version=SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionError) as excinfo:
        load_session(session, data_cfg.columns)
    assert "regenerate" in str(excinfo.value).lower()


def test_a_manifest_listing_an_absent_file_is_rejected(
    tmp_path: Path, frame: pd.DataFrame, data_cfg: DataConfig
) -> None:
    """The cross-check whose absence let the committed reference run rot unnoticed."""
    session = _minimal_session(tmp_path, frame, data_cfg)
    payload = json.loads((session / MANIFEST_FILE).read_text(encoding="utf-8"))
    payload["artifacts"].append({"role": "best_model", "filename": "ghost.keras"})
    (session / MANIFEST_FILE).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError) as excinfo:
        load_session(session, data_cfg.columns)
    assert "ghost.keras" in str(excinfo.value)


def test_no_filename_literal_appears_in_the_dashboard_package() -> None:
    """FR-020: names resolve through manifest roles, never through a literal.

    Asserted mechanically because this is a rule a future edit breaks silently — and
    it is the exact bug `session.py` was created to prevent.
    """
    package = REPO_ROOT / "apps" / "backend" / "src" / "mlpp" / "dashboard"
    offenders: list[str] = []
    for path in package.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]
            if any(suffix in code for suffix in (".keras", ".gz", "manifest.json")):
                offenders.append(f"{path.relative_to(package)}:{number}: {line.strip()}")
    assert not offenders, "artifact filenames must resolve through the manifest:\n" + "\n".join(
        offenders
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not (EXAMPLE_SESSION / MANIFEST_FILE).is_file(),
    reason="committed reference session is absent",
)
def test_introspect_renders_against_the_committed_session() -> None:
    """Import-and-call, exercising the panel's real data path outside a browser.

    Streamlit's API is callable with no script run context — it warns and no-ops — so
    this catches signature drift, manifest-role typos and unhandled shapes, which is
    what a panel test can honestly cover without a browser.
    """
    from mlpp.dashboard.loaders import columns_for
    from mlpp.dashboard.panels import introspect
    from mlpp.predict import load_model

    session = load_session(EXAMPLE_SESSION, columns_for(str(EXAMPLE_SESSION)))
    introspect.render(session, load_model(session))


@pytest.mark.slow
@pytest.mark.skipif(
    not (EXAMPLE_SESSION / MANIFEST_FILE).is_file(),
    reason="committed reference session is absent",
)
def test_the_app_renders_every_panel_for_the_committed_session() -> None:
    """Drive the real entrypoint headlessly, through Streamlit's own harness.

    Covers what panel-level tests cannot: that `app.py` wires the sidebar, the
    loaders and the panels together, and that the default OUTPUTS path finds real
    sessions. Marked slow — it loads the Keras model.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_SCRIPT), default_timeout=300).run()

    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert [s.value for s in app.subheader] == [
        "Architecture",
        "Feature contract",
        "Training history",
        "Test metrics",
        "Artifacts",
    ]
    assert any("example" in str(option) for option in app.selectbox[0].options)


@pytest.mark.slow
def test_an_invalid_session_renders_an_error_not_a_traceback(tmp_path: Path) -> None:
    """FR-002 through the real entrypoint: a message naming the problem, no traceback."""
    from streamlit.testing.v1 import AppTest

    (tmp_path / "2026-01-01_00-00-00").mkdir()

    app = AppTest.from_file(str(APP_SCRIPT), default_timeout=300).run()
    app = app.text_input[0].set_value(str(tmp_path)).run()
    app = app.selectbox[0].select(app.selectbox[0].options[0]).run()

    assert not app.exception, "an invalid session must never surface a traceback"
    assert app.error, "expected a rendered error naming the problem"
    assert MANIFEST_FILE in app.error[0].value


def _minimal_session(
    tmp_path: Path,
    frame: pd.DataFrame,
    data_cfg: DataConfig,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> Path:
    """A session directory with a real fitted state and a hand-written manifest."""
    from mlpp.preprocess import Preprocessor
    from mlpp.session import SessionWriter, write_fitted_state

    writer = SessionWriter(tmp_path / "session")
    pre = Preprocessor(data_cfg.columns).fit(frame)
    write_fitted_state(writer, pre.fitted_state)

    manifest = {
        "schema_version": schema_version,
        "created": "2026-08-04T00:00:00",
        "features": {
            "numeric_columns": list(pre.schema.numeric),
            "categorical_columns": list(pre.schema.categorical),
            "output_column": pre.schema.output,
            "feature_names": list(pre.feature_names),
        },
        "artifacts": [
            {"role": role, "filename": name}
            for role, name in (
                ("scaler", "scaler.gz"),
                ("output_scaler", "output_scaler.gz"),
                ("encoders", "encoders.gz"),
            )
        ],
    }
    (writer.session_dir / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    return writer.session_dir

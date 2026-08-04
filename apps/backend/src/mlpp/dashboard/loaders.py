"""Every artifact read and every cache decision, in one place.

Streamlit reruns the whole script on each interaction, so uncached loading would
re-read the Keras model on every click. Two decorators, chosen by what they hold:

- `@st.cache_resource` for the Keras model — unhashable, one live object per session
  directory.
- `@st.cache_data` for dataframes and importance results — copied per caller.

Cache keys are scalars only (`session_dir`, `dataset_path`, `seed`, `n_repeats`).
Keying on a mutable object is how a cache starts returning the wrong session.

`session_dir` is typed as `str` rather than `Path` throughout: Streamlit hashes the
key, and a plain string is unambiguous to hash and to display.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:  # `keras` is imported lazily below — this is for the type only.
    import keras

from mlpp.config import ColumnConfig
from mlpp.data import read_table_auto
from mlpp.importance import ImportanceResult, permutation_importance
from mlpp.session import LoadedSession, SessionManifest, load_session, read_manifest

#: loaders -> dashboard -> mlpp -> src -> backend -> apps -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[5]


def default_outputs_root() -> Path:
    """Where `mlpp-train` writes sessions, as the sidebar's starting value.

    Lives here rather than in `app.py` because that module runs the app on import,
    which makes anything defined there untestable. Getting this depth wrong is silent:
    the app renders an empty "no sessions found" page rather than failing.
    """
    return _REPO_ROOT / "OUTPUTS"


def list_sessions(root: Path) -> list[Path]:
    """Candidate session directories under `root`, newest first.

    A directory is a *candidate* if it exists — not if it is valid. Validation is
    `load_session`'s job, and its failure is what the error surface renders. Filtering
    invalid sessions out here would hide exactly the directories the author most needs
    told about.

    Sorted by name descending, which is chronological because session directories are
    timestamped (`session.SESSION_FORMAT`). A missing root yields an empty list rather
    than raising: an empty OUTPUTS/ is a legitimate starting state, not an error.
    """
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)


def columns_for(session_dir: str) -> ColumnConfig:
    """Rebuild the column contract from the session's own manifest.

    The manifest is the single source of truth for the column contract, so nothing
    here restates what the model was trained on.
    """
    features = read_manifest(Path(session_dir)).features
    return ColumnConfig(
        input_columns=features.numeric_columns + features.categorical_columns,
        output_column=features.output_column,
        categorical_columns=features.categorical_columns,
    )


@st.cache_data(show_spinner="Reading session…")
def load_manifest_cached(session_dir: str) -> SessionManifest:
    return read_manifest(Path(session_dir))


@st.cache_resource(show_spinner="Loading session…")
def load_session_cached(session_dir: str) -> LoadedSession:
    """A restored session: manifest plus a fitted Preprocessor.

    `cache_resource` rather than `cache_data` because a LoadedSession holds fitted
    sklearn estimators, which are not meaningfully copyable per caller.
    """
    return load_session(Path(session_dir), columns_for(session_dir))


@st.cache_resource(show_spinner="Loading model… (first load imports TensorFlow)")
def load_model_cached(session_dir: str) -> keras.Model:
    """The session's Keras model. `cache_resource`: unhashable and expensive.

    `mlpp.predict` is imported inside the function so that merely opening the app
    does not pay the multi-second TensorFlow import before a session is chosen.
    """
    from mlpp.predict import load_model

    return load_model(load_session_cached(session_dir))


@st.cache_data(show_spinner="Reading dataset…")
def load_dataset_cached(dataset_path: str) -> pd.DataFrame:
    return read_table_auto(Path(dataset_path))


@st.cache_data(show_spinner=False)
def compute_importance_cached(
    session_dir: str, dataset_path: str, seed: int, n_repeats: int
) -> ImportanceResult:
    """Permutation importance, keyed on four scalars (FR-010).

    The `progress` callback is deliberately *not* a parameter: a callable is not a
    stable cache key, and passing one would either defeat the cache or key it on a
    bound method's identity. The caller draws progress around this call instead.
    """
    from mlpp.predict import make_scorer

    session = load_session_cached(session_dir)
    return permutation_importance(
        session.preprocessor,
        make_scorer(load_model_cached(session_dir)),
        load_dataset_cached(dataset_path),
        n_repeats=n_repeats,
        seed=seed,
    )

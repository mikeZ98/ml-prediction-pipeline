"""Ranked permutation feature importance. FR-008 through FR-010.

Consumes `mlpp.importance`; computes nothing itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from mlpp.dashboard import loaders
from mlpp.errors import MlppError
from mlpp.importance import ImportanceResult

if TYPE_CHECKING:  # for the annotation only; the caller supplies a loaded model.
    import keras

from mlpp.session import LoadedSession


def render(session: LoadedSession, model: keras.Model) -> None:
    """Pick a dataset, run importance against it, read the result as a ranked chart."""
    st.subheader("Feature importance")
    st.caption(
        "How much test R² drops when each input column is shuffled. Higher means the "
        "model relies on it more."
    )

    dataset_path = _dataset_picker(session)
    if dataset_path is None:
        return

    left, right = st.columns(2)
    n_repeats = left.number_input(
        "Repeats", min_value=1, max_value=50, value=5, help="Shuffles per column."
    )
    seed = right.number_input("Seed", min_value=0, max_value=10_000, value=0)

    if not st.button("Compute importance", type="primary"):
        st.info("Choose a dataset and press Compute.")
        return

    # Progress is drawn around the cached call rather than passed into it: a callable
    # is not a stable cache key, so threading one through would defeat the cache.
    bar = st.progress(0.0, text="Scoring permutations…")
    try:
        result = loaders.compute_importance_cached(
            str(session.session_dir), str(dataset_path), int(seed), int(n_repeats)
        )
    except MlppError as exc:
        bar.empty()
        st.error(str(exc))
        return
    bar.progress(1.0, text="Done")
    bar.empty()

    _chart(result)


def _dataset_picker(session: LoadedSession) -> Path | None:
    """Datasets to score against. Importance needs the target column, so TEST/ is the
    natural source — but any CSV or Parquet the author names is allowed."""
    default_dir = loaders.default_outputs_root().parent / "TEST"
    directory = Path(
        st.text_input(
            "Dataset directory",
            value=str(default_dir),
            key="importance_dir",
            help="Importance needs a target column, so use a labelled dataset.",
        )
    )
    if not directory.is_dir():
        st.warning(f"Not a directory: {directory}")
        return None

    candidates = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".csv", ".parquet", ".pq"}
    )
    if not candidates:
        st.warning(f"No CSV or Parquet files in {directory}")
        return None

    return st.selectbox(
        "Dataset", options=candidates, format_func=lambda p: p.name, key="importance_dataset"
    )


def _chart(result: ImportanceResult) -> None:
    st.metric("Baseline R²", f"{result.baseline_r2:.4f}")
    st.caption(f"{result.n_repeats} repeats · seed {result.seed}")

    frame = pd.DataFrame(
        {
            "column": [c.column for c in result.columns],
            "mean R² drop": [c.mean_drop for c in result.columns],
            "std": [c.std_drop for c in result.columns],
        }
    ).set_index("column")

    # Horizontal bars, most important at the top. Negative values are plotted as-is:
    # a column the model ignores can score below zero because shuffling happened to
    # help, and that sign is the evidence it is unused.
    st.bar_chart(frame["mean R² drop"], horizontal=True)
    st.dataframe(
        frame.reset_index(),
        width="stretch",
        hide_index=True,
        column_config={
            "mean R² drop": st.column_config.NumberColumn(format="%.5f"),
            "std": st.column_config.NumberColumn(format="%.5f"),
        },
    )
    if any(c.mean_drop < 0 for c in result.columns):
        st.caption(
            "Negative values mean shuffling the column happened to improve the score — "
            "evidence the model does not use it. Shown unclamped on purpose."
        )

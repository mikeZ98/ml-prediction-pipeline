"""Score one hand-entered row. FR-012, FR-013, FR-022.

Fields are generated from the session's own feature contract — no column list is
ever hardcoded here, because the manifest is the single source of truth for it.
Writes nothing to disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from mlpp.errors import MlppError
from mlpp.preprocess import flag_out_of_range

if TYPE_CHECKING:  # for the annotation only; the caller supplies a loaded model.
    import keras

from mlpp.session import LoadedSession

#: How far outside the fitted distribution an entry may sit before it is flagged.
MAX_SIGMA = 3.0


def render(session: LoadedSession, model: keras.Model) -> None:
    st.subheader("Single-row inference")
    st.caption("Enter one row of inputs and score it. Nothing is written to disk.")

    features = session.manifest.features
    categories = session.preprocessor.fitted_categories

    with st.form("single_row"):
        values = _input_fields(session, features.numeric_columns, categories)
        submitted = st.form_submit_button("Score", type="primary")

    if not submitted:
        return

    warnings = flag_out_of_range(session.preprocessor, _numeric_only(values), max_sigma=MAX_SIGMA)
    for warning in warnings:
        st.warning(
            f"**{warning.column}** = {warning.value:g} is {warning.sigma:+.1f}σ from the "
            f"fitted mean. Training data spanned roughly {warning.low:g} to {warning.high:g}. "
            "The model will still return a number, but it is extrapolating."
        )

    try:
        from mlpp.predict import score_frame

        result = score_frame(session, model, pd.DataFrame([values]))
    except MlppError as exc:
        st.error(str(exc))
        return

    st.metric(features.output_column, f"{float(result.predictions[0]):.6g}")
    if result.has_unseen:
        st.warning(result.describe_unseen())


def _input_fields(
    session: LoadedSession,
    numeric_columns: tuple[str, ...],
    categories: dict[str, tuple[object, ...]],
) -> dict[str, Any]:
    """One widget per input column, generated from the contract.

    Numeric fields default to the scaler's fitted mean rather than zero: it is a
    valid, in-distribution starting point, so the form does not open already
    triggering its own range warning.
    """
    means = _fitted_means(session, numeric_columns)
    values: dict[str, Any] = {}

    columns = st.columns(2)
    for index, column in enumerate(numeric_columns):
        values[column] = columns[index % 2].number_input(
            column, value=float(means.get(column, 0.0)), format="%g", key=f"single_{column}"
        )

    for index, (column, levels) in enumerate(sorted(categories.items())):
        values[column] = columns[index % 2].selectbox(
            column, options=list(levels), key=f"single_cat_{column}"
        )
    return values


def _fitted_means(session: LoadedSession, numeric_columns: tuple[str, ...]) -> dict[str, float]:
    """Fitted mean per numeric column, or an empty mapping if state is unavailable."""
    try:
        scaler = session.preprocessor.fitted_state.scaler
    except MlppError:
        return {}
    return {
        column: float(mean)
        for column, mean in zip(session.preprocessor.schema.numeric, scaler.mean_, strict=False)
        if column in numeric_columns
    }


def _numeric_only(values: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}

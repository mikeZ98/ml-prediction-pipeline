"""Score a whole CSV or Parquet file, with a download. FR-014 through FR-016.

Nothing is written into OUTPUTS/ — the result leaves through `st.download_button`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from mlpp.dashboard import loaders
from mlpp.errors import MlppError
from mlpp.predict_cli import PREDICTION_COLUMN, prediction_frame

if TYPE_CHECKING:  # for the annotation only; the caller supplies a loaded model.
    import keras

from mlpp.session import LoadedSession

_SUFFIXES = {".csv", ".parquet", ".pq"}


def render(session: LoadedSession, model: keras.Model) -> None:
    st.subheader("Batch inference")
    st.caption("Score a CSV or Parquet file. The session directory is never written to.")

    source = _source_picker()
    if source is None:
        return

    keep_inputs = st.toggle(
        "Include input columns in the download",
        value=False,
        help="Off: the prediction column alone. On: the inputs beside it.",
    )

    if not st.button("Score file", type="primary", key="batch_score"):
        st.info("Choose a file and press Score.")
        return

    try:
        frame = loaders.load_dataset_cached(str(source))
        from mlpp.predict import score_frame

        result = score_frame(session, model, frame)
    except MlppError as exc:
        st.error(str(exc))
        return

    # FR-016. The CLI emits this only as a log line; here it is a visible warning,
    # because an all-zeros encoding still produces a confident-looking number.
    if result.has_unseen:
        st.warning(result.describe_unseen())

    predictions = pd.Series(result.predictions, index=frame.index, name=PREDICTION_COLUMN)
    out = prediction_frame(frame, predictions, keep_inputs=keep_inputs)

    st.success(f"Scored {len(out):,} row(s).")
    st.dataframe(out.head(200), width="stretch", hide_index=True)
    if len(out) > 200:
        st.caption(f"Previewing the first 200 of {len(out):,} rows. The download has them all.")

    st.download_button(
        "Download predictions (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"{source.stem}_predictions.csv",
        mime="text/csv",
    )


def _source_picker() -> Path | None:
    directory = Path(
        st.text_input(
            "Input directory",
            value=str(loaders.default_outputs_root().parent / "TEST"),
            key="batch_dir",
        )
    )
    if not directory.is_dir():
        st.warning(f"Not a directory: {directory}")
        return None

    candidates = sorted(p for p in directory.iterdir() if p.suffix.lower() in _SUFFIXES)
    if not candidates:
        st.warning(f"No CSV or Parquet files in {directory}")
        return None

    return st.selectbox("File", options=candidates, format_func=lambda p: p.name, key="batch_file")

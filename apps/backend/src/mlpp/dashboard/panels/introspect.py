"""Read-only view of a trained session: architecture, contract, history, metrics.

FR-004 through FR-007. Every filename is resolved through
`manifest.filenames_for(ROLE_*)` — spelling one here would be the same bug
`session.py` exists to prevent. Nothing in this module writes.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:  # for the annotation only; the caller supplies a loaded model.
    import keras

from mlpp.session import (
    ROLE_METRICS,
    ROLE_STAGE_HISTORY,
    ROLE_TRAINING_LOG,
    LoadedSession,
    SessionManifest,
)


def render(session: LoadedSession, model: keras.Model) -> None:
    """Draw every introspection surface for an already-loaded session."""
    _architecture(model)
    st.divider()
    _feature_contract(session)
    st.divider()
    _training_history(session)
    st.divider()
    _test_metrics(session)
    st.divider()
    _inventory(session)


def _architecture(model: keras.Model) -> None:
    """FR-004. `model.summary()` prints; capture it rather than letting it hit stdout."""
    st.subheader("Architecture")

    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    st.code(buffer.getvalue(), language="text")

    total = int(sum(int(w.size) for w in model.get_weights()))
    left, right = st.columns(2)
    left.metric("Input shape", str(model.input_shape))
    right.metric("Parameters", f"{total:,}")


def _feature_contract(session: LoadedSession) -> None:
    """FR-005. The manifest is the single source of truth for the column contract."""
    st.subheader("Feature contract")
    features = session.manifest.features

    left, right = st.columns(2)
    left.metric("Input columns", len(features.numeric_columns) + len(features.categorical_columns))
    right.metric("Model features (post one-hot)", len(features.feature_names))

    st.caption(f"Target: `{features.output_column}`")
    st.dataframe(
        pd.DataFrame(
            {
                "column": list(features.numeric_columns) + list(features.categorical_columns),
                "kind": ["numeric"] * len(features.numeric_columns)
                + ["categorical"] * len(features.categorical_columns),
            }
        ),
        width="stretch",
        hide_index=True,
    )

    if len(features.feature_names) != len(features.numeric_columns) + len(
        features.categorical_columns
    ):
        with st.expander("Expanded feature order (axis 1 of X)"):
            st.write(list(features.feature_names))


def _training_history(session: LoadedSession) -> None:
    """FR-006. Per-stage history plus the aggregate training log, whichever exist."""
    st.subheader("Training history")

    stages = _frames_for(session, ROLE_STAGE_HISTORY)
    if not stages:
        st.info("This session records no per-stage training history.")
    for filename, frame in stages:
        st.caption(filename)
        numeric = frame.select_dtypes("number")
        if not numeric.empty:
            st.line_chart(numeric)
        st.dataframe(frame, width="stretch", hide_index=True)

    for filename, frame in _frames_for(session, ROLE_TRAINING_LOG):
        st.caption(filename)
        st.dataframe(frame, width="stretch", hide_index=True)


def _test_metrics(session: LoadedSession) -> None:
    """FR-007."""
    st.subheader("Test metrics")
    frames = _frames_for(session, ROLE_METRICS)
    if not frames:
        st.info("This session records no test metrics.")
    for filename, frame in frames:
        st.caption(filename)
        st.dataframe(frame, width="stretch", hide_index=True)


def _inventory(session: LoadedSession) -> None:
    """Every artifact the manifest claims, with on-disk presence confirmed.

    `load_session` already refuses a session whose manifest lists an absent file, so
    in practice every row here is present. It is shown anyway because this table is
    the thing an author reads when a session looks wrong.
    """
    st.subheader("Artifacts")
    rows = [
        {
            "role": entry.role,
            "filename": entry.filename,
            "on disk": (session.session_dir / entry.filename).is_file(),
            "size": _size_of(session.session_dir / entry.filename),
        }
        for entry in session.manifest.artifacts
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"schema_version {session.manifest.schema_version} · created {session.manifest.created}"
    )


def _frames_for(session: LoadedSession, role: str) -> list[tuple[str, pd.DataFrame]]:
    """Every CSV recorded under `role`, resolved through the manifest.

    An unreadable artifact is skipped rather than raised: one malformed history CSV
    should not blank the whole page, and the inventory table above already reports
    what is present.
    """
    frames: list[tuple[str, pd.DataFrame]] = []
    for filename in _filenames(session.manifest, role):
        path = session.session_dir / filename
        if not path.is_file():
            continue
        try:
            frames.append((filename, pd.read_csv(path)))
        except (OSError, ValueError):
            continue
    return frames


def _filenames(manifest: SessionManifest, role: str) -> tuple[str, ...]:
    return manifest.filenames_for(role)


def _size_of(path: Path) -> str:
    if not path.is_file():
        return "—"
    size = float(path.stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"

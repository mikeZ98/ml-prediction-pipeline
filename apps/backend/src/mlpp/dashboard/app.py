"""Dashboard entrypoint.

Run from `apps/backend/`:

    uv run --group dashboard streamlit run src/mlpp/dashboard/app.py

Owns page config, the sidebar session selector, and dispatch to panels. Holds no
domain logic. Changing the selected session re-derives every panel through
Streamlit's natural rerun — there is no manual cache invalidation, because every
cache key already carries the session directory (FR-003).

`MlppError` is caught here, at the panel boundary, and rendered with `st.error`
(FR-002). That catch belongs in exactly one place; a traceback must never reach the
browser.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mlpp.dashboard import loaders
from mlpp.dashboard.panels import batch, importance, introspect, single
from mlpp.errors import MlppError

#: Tab label -> the panel that draws it. One entry per PRD surface; adding a surface
#: means adding a module and a line here, never branching inside a panel.
PANELS = (
    ("Session", introspect.render),
    ("Importance", importance.render),
    ("Single row", single.render),
    ("Batch", batch.render),
)


def main() -> None:
    st.set_page_config(page_title="mlpp — session explorer", page_icon="📈", layout="wide")
    st.title("mlpp session explorer")

    session_dir = _sidebar_session_picker()
    if session_dir is None:
        return

    # One catch for the whole page. An invalid session is the expected case here —
    # OUTPUTS/ accumulates interrupted runs — so it gets a message naming the problem
    # rather than a stack trace.
    try:
        session = loaders.load_session_cached(str(session_dir))
        model = loaders.load_model_cached(str(session_dir))
    except MlppError as exc:
        st.error(str(exc))
        st.caption(
            "Artifacts are rejected rather than migrated. Regenerate the session with "
            "`uv run mlpp-train`."
        )
        return

    # Every panel is wrapped, not just the loaders above: a panel that raises mid-render
    # must not take the whole page down with a traceback.
    for tab, (_, render) in zip(st.tabs([label for label, _ in PANELS]), PANELS, strict=True):
        with tab:
            try:
                render(session, model)
            except MlppError as exc:
                st.error(str(exc))


def _sidebar_session_picker() -> Path | None:
    """Choose a session directory. Returns None when there is nothing to show."""
    st.sidebar.header("Session")

    root = Path(
        st.sidebar.text_input(
            "OUTPUTS directory",
            value=str(loaders.default_outputs_root()),
            help="Where mlpp-train writes its timestamped session directories.",
        )
    )

    candidates = loaders.list_sessions(root)
    if not candidates:
        st.sidebar.warning("No session directories found.")
        st.info(
            f"No sessions under `{root}`. Train one with `uv run mlpp-train`, or point the "
            "sidebar at a different directory."
        )
        return None

    chosen = st.sidebar.selectbox(
        "Directory",
        options=candidates,
        format_func=lambda p: p.name,
        help="Newest first. Invalid sessions are listed too — selecting one reports why.",
    )
    st.sidebar.caption(f"{len(candidates)} session(s) found")
    return chosen


main()

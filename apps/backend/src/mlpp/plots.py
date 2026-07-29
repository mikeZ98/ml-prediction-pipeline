"""Artifact plotting. Isolated here so metrics stay importable without a GUI stack."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window from a pipeline run
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from mlpp.metrics import rolling_error  # noqa: E402


def plot_training_curves(history: Mapping[str, Sequence[float]], path: Path) -> Path:
    """Write loss and MAE curves for one training stage to `path`.

    Takes a full path rather than a directory + tag: naming files in a session
    directory is `session.py`'s job, not this module's.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_mae, "mae", "MAE")):
        for prefix, label in (("", "train"), ("val_", "val")):
            values = history.get(f"{prefix}{key}")
            if values:
                ax.plot(values, label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    tag: str,
    window: int,
    active_mask: np.ndarray | None = None,
) -> Path:
    """Write the interactive truth-vs-prediction / residual report to `path`.

    `tag` survives only as the chart title; the filename comes from the caller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    diffs = pred - true
    roll = rolling_error(true, pred, window, active_mask)
    roll_label = f"rolling({window})" + (" [active]" if active_mask is not None else "")

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Truth vs Pred", "Diff + Rolling"))
    fig.add_trace(go.Scatter(y=true, mode="lines", name="true"), row=1, col=1)
    fig.add_trace(go.Scatter(y=pred, mode="lines", name="pred"), row=1, col=1)
    fig.add_trace(go.Scatter(y=diffs, mode="lines", name="diff"), row=2, col=1)
    fig.add_trace(go.Scatter(y=roll, mode="lines", name=roll_label), row=2, col=1)
    fig.update_layout(height=800, title_text=f"Prediction analysis — {tag}", showlegend=True)

    fig.write_html(str(path))
    return path

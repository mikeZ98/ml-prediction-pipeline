"""Scoring a trained session: the reader side of the artifact contract.

`session.py` restores the manifest and a fitted `Preprocessor` but deliberately
stops short of the model. This module adds it: the session's ONNX graph, executed
by ONNX Runtime through `mlpp.inference.onnx_model`. Nothing here imports the
training framework, so the predict CLI, the dashboard and the API all score with
the `train` extra absent.

Loading and scoring are separate calls on purpose: building an inference session
costs far more than one forward pass, so a long-lived caller (a service) pays it
once and scores many frames.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlpp.inference.onnx_model import OnnxModel, load_onnx_model
from mlpp.session import LoadedSession


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Predictions in engineering units, plus what the input got wrong.

    `unseen` is non-empty when the input carried categorical levels the encoder
    never saw. Those rows still produce a number — the encoder zero-fills them —
    which is exactly why they are reported rather than swallowed.
    """

    predictions: np.ndarray
    #: Categorical column -> the unrecognised values found in it.
    unseen: dict[str, tuple[object, ...]]
    #: How many input rows carried at least one unrecognised level.
    unseen_rows: int

    @property
    def has_unseen(self) -> bool:
        return bool(self.unseen)

    def describe_unseen(self) -> str:
        """One-line summary for a CLI or log. Empty string when nothing is unseen."""
        if not self.unseen:
            return ""
        parts = [f"{column}={list(values)}" for column, values in sorted(self.unseen.items())]
        return (
            f"{self.unseen_rows} row(s) carried categorical levels not seen during training "
            f"and were encoded as all-zeros: {'; '.join(parts)}"
        )


def load_model(session: LoadedSession, *, intra_op_threads: int | None = None) -> OnnxModel:
    """Load the session's ONNX model, resolving its filename through the manifest."""
    return load_onnx_model(session, intra_op_threads=intra_op_threads)


def make_scorer(model: OnnxModel) -> Callable[[np.ndarray], np.ndarray]:
    """Bind `model` into the `Scorer` callable `mlpp.importance` expects.

    Supplying the closure here is what lets `importance.py` stay runtime-free.

    Returns *scaled* predictions deliberately. `permutation_importance` compares them
    against the scaled `y` from `transform`, and R² is invariant under that shared
    affine transform — so inverting would cost a round-trip per permutation and
    change nothing.
    """

    def score(x: np.ndarray) -> np.ndarray:
        return model.predict(x)

    return score


def score_frame(session: LoadedSession, model: OnnxModel, df: pd.DataFrame) -> PredictionResult:
    """Score `df` against a loaded session, returning predictions in engineering units.

    Mirrors the training-time evaluation path: transform, predict, then invert the
    target scaling. The target column is optional — inference input rarely has one.
    """
    preprocessor = session.preprocessor
    unseen, unseen_rows = _unseen_levels(preprocessor.fitted_categories, df)

    x, _ = preprocessor.transform(df)
    predictions = preprocessor.inverse_target(model.predict(x))
    return PredictionResult(predictions=predictions, unseen=unseen, unseen_rows=unseen_rows)


def _unseen_levels(
    fitted: dict[str, tuple[object, ...]], df: pd.DataFrame
) -> tuple[dict[str, tuple[object, ...]], int]:
    """Find categorical values absent from the fitted levels, and how many rows hold one.

    Nulls are excluded: `transform` forward/backward-fills them, so a NaN is a
    filled value rather than an unrecognised level.
    """
    report: dict[str, tuple[object, ...]] = {}
    affected = pd.Series(False, index=df.index)

    for column, known in fitted.items():
        if column not in df.columns:
            continue
        values = df[column]
        offending = values.notna() & ~values.isin(set(known))
        if not bool(offending.any()):
            continue
        report[column] = tuple(sorted(values[offending].unique(), key=repr))
        affected |= offending

    return report, int(affected.sum())

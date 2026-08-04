"""Scoring a trained session: the reader side of the artifact contract.

`session.py` restores the manifest and a fitted `Preprocessor` but deliberately
stops short of the model — loading Keras there would make inspecting or validating
a session require the whole training stack. This module is where that line is
crossed, and it is the only new module in the package that imports Keras.

Loading and scoring are separate calls on purpose: importing TensorFlow costs
seconds, so a long-lived caller (a service) pays it once and scores many frames.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd

from mlpp.errors import ArtifactError
from mlpp.session import ROLE_BEST_MODEL, LoadedSession


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


def load_model(session: LoadedSession) -> keras.Model:
    """Load the session's best model, resolving its filename through the manifest.

    The filename is never spelled here — `session.py` owns every name in a session
    directory, so this asks the manifest inventory for the `best_model` role.
    """
    filenames = session.manifest.filenames_for(ROLE_BEST_MODEL)
    if not filenames:
        raise ArtifactError(
            f"{session.session_dir}: manifest records no {ROLE_BEST_MODEL!r} artifact; "
            "the session has no model to score with"
        )
    if len(filenames) > 1:
        raise ArtifactError(
            f"{session.session_dir}: manifest records {len(filenames)} {ROLE_BEST_MODEL!r} "
            f"artifacts ({list(filenames)}); expected exactly one"
        )
    return keras.saving.load_model(session.session_dir / filenames[0])


def make_scorer(model: keras.Model) -> Callable[[np.ndarray], np.ndarray]:
    """Bind `model` into the `Scorer` callable `mlpp.importance` expects.

    This module already owns the Keras import, so supplying the closure here is what
    lets `importance.py` stay TensorFlow-free.

    Returns *scaled* predictions deliberately. `permutation_importance` compares them
    against the scaled `y` from `transform`, and R² is invariant under that shared
    affine transform — so inverting would cost a round-trip per permutation and
    change nothing.
    """

    def score(x: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(x, verbose=0))

    return score


def score_frame(session: LoadedSession, model: keras.Model, df: pd.DataFrame) -> PredictionResult:
    """Score `df` against a loaded session, returning predictions in engineering units.

    Mirrors the training-time evaluation path: transform, predict, then invert the
    target scaling. The target column is optional — inference input rarely has one.
    """
    preprocessor = session.preprocessor
    unseen, unseen_rows = _unseen_levels(preprocessor.fitted_categories, df)

    x, _ = preprocessor.transform(df)
    predictions = preprocessor.inverse_target(model.predict(x, verbose=0))
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

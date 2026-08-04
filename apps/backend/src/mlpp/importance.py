"""Global permutation feature importance. No TensorFlow import — keep it cheap.

How much does the model's accuracy degrade when one input column's values are
shuffled? The column whose loss hurts most matters most.

Two design choices carry this module:

Scoring is injected as a `Scorer` callable rather than taking a Keras model, so
nothing here imports TensorFlow and the whole loop is exercisable in the fast suite
with a stub. `mlpp.predict.make_scorer` supplies the real one.

Permutation happens on the *raw DataFrame column*, before `transform`. One-hot
expansion runs after `align_columns` (`preprocess.py`), so shuffling a source column
reshuffles all of that column's one-hot positions together by construction. That is
what makes a categorical column report as one number instead of one per level —
aggregation is structural rather than a post-hoc regroup over `feature_names`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlpp.errors import SchemaError
from mlpp.metrics import regression_metrics
from mlpp.preprocess import Preprocessor

#: Maps X of shape (rows, n_features, 1) to raw predictions. Scaled units are fine:
#: R² is invariant under the shared affine transform, so the baseline and permuted
#: scores stay comparable without an inverse round-trip per permutation.
Scorer = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class ColumnImportance:
    """One input column's degradation across repeats."""

    #: An *input* column, not a one-hot expansion of one.
    column: str
    #: Mean R² degradation. Negative is meaningful — see `permutation_importance`.
    mean_drop: float
    std_drop: float
    scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ImportanceResult:
    baseline_r2: float
    #: Sorted by `mean_drop`, descending.
    columns: tuple[ColumnImportance, ...]
    n_repeats: int
    seed: int


def permutation_importance(
    preprocessor: Preprocessor,
    scorer: Scorer,
    frame: pd.DataFrame,
    *,
    n_repeats: int = 5,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> ImportanceResult:
    """Rank `preprocessor.schema.inputs` by R² lost when each is shuffled.

    Each repeat starts from a *fresh copy* of the original frame. Shuffling in place
    would accumulate: every column would be measured on top of every previously
    permuted one, yielding the joint degradation rather than each column's marginal
    contribution — a monotonically rising curve that looks plausible and is wrong.

    Negative values are returned unclamped. A column the model ignores can score
    slightly below zero because shuffling it happened to help, and that sign is the
    evidence the feature is unused; clamping would erase it.

    `progress` is called as `(completed, total)` so a caller can draw a bar without
    this module importing any interface library.

    Raises SchemaError when `frame` carries no target column — importance is
    undefined without one to score against.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    baseline_x, baseline_y = preprocessor.transform(frame)
    if baseline_y is None:
        raise SchemaError(
            f"target column {preprocessor.schema.output!r} is absent from the frame; "
            "permutation importance needs a target to score against"
        )
    baseline_r2 = regression_metrics(baseline_y, scorer(baseline_x)).r2

    columns = preprocessor.schema.inputs
    total = len(columns) * n_repeats
    completed = 0
    results: list[ColumnImportance] = []

    for column in columns:
        drops: list[float] = []
        for repeat in range(n_repeats):
            rng = np.random.default_rng(seed + repeat)
            permuted = frame.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())

            x, y = preprocessor.transform(permuted)
            # `y` cannot be None here: the same frame produced a target above, and
            # permuting one input column never removes it.
            assert y is not None
            drops.append(baseline_r2 - regression_metrics(y, scorer(x)).r2)

            completed += 1
            if progress is not None:
                progress(completed, total)

        results.append(
            ColumnImportance(
                column=column,
                mean_drop=float(np.mean(drops)),
                std_drop=float(np.std(drops)),
                scores=tuple(drops),
            )
        )

    results.sort(key=lambda c: c.mean_drop, reverse=True)
    return ImportanceResult(
        baseline_r2=baseline_r2,
        columns=tuple(results),
        n_repeats=n_repeats,
        seed=seed,
    )

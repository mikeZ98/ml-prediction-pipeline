"""Permutation importance.

Almost all of this runs without TensorFlow: the loop is parameterised over a scorer
callable, so a stub with known behaviour pins the algorithm's properties in the fast
suite. Only the genuine Keras path is marked `slow`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlpp.config import ColumnConfig
from mlpp.errors import SchemaError
from mlpp.importance import permutation_importance
from mlpp.preprocess import Preprocessor

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SESSION = REPO_ROOT / "OUTPUTS" / "example"


@pytest.fixture
def fitted(frame: pd.DataFrame, column_cfg: ColumnConfig) -> Preprocessor:
    return Preprocessor(column_cfg).fit(frame)


def _reads_feature(index: int):
    """A scorer whose output depends on exactly one position of X."""

    def score(x: np.ndarray) -> np.ndarray:
        return x[:, index, 0]

    return score


def _constant(x: np.ndarray) -> np.ndarray:
    return np.zeros(x.shape[0])


def test_only_the_consulted_column_ranks_high(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    """A scorer reading one feature must rank that column top and the rest near zero."""
    target = "feature_02"
    result = permutation_importance(
        fitted, _reads_feature(fitted.index_of(target)), frame, n_repeats=3
    )

    assert result.columns[0].column == target
    others = [c.mean_drop for c in result.columns if c.column != target]
    assert result.columns[0].mean_drop > max(others)
    assert all(abs(drop) < 1e-9 for drop in others)


def test_a_constant_scorer_finds_nothing_important(
    fitted: Preprocessor, frame: pd.DataFrame
) -> None:
    result = permutation_importance(fitted, _constant, frame, n_repeats=2)
    assert all(abs(c.mean_drop) < 1e-9 for c in result.columns)


def test_results_are_sorted_by_mean_drop_descending(
    fitted: Preprocessor, frame: pd.DataFrame
) -> None:
    result = permutation_importance(
        fitted, _reads_feature(fitted.index_of("feature_01")), frame, n_repeats=2
    )
    drops = [c.mean_drop for c in result.columns]
    assert drops == sorted(drops, reverse=True)


def test_a_fixed_seed_reproduces_identical_numbers(
    fitted: Preprocessor, frame: pd.DataFrame
) -> None:
    scorer = _reads_feature(fitted.index_of("feature_01"))
    first = permutation_importance(fitted, scorer, frame, n_repeats=3, seed=7)
    second = permutation_importance(fitted, scorer, frame, n_repeats=3, seed=7)

    assert [c.scores for c in first.columns] == [c.scores for c in second.columns]
    assert first.baseline_r2 == second.baseline_r2


def test_a_different_seed_shuffles_differently(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    scorer = _reads_feature(fitted.index_of("feature_01"))
    first = permutation_importance(fitted, scorer, frame, n_repeats=3, seed=0)
    other = permutation_importance(fitted, scorer, frame, n_repeats=3, seed=99)
    assert [c.scores for c in first.columns] != [c.scores for c in other.columns]


def test_n_repeats_controls_the_number_of_scores(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    result = permutation_importance(fitted, _constant, frame, n_repeats=4)
    assert result.n_repeats == 4
    assert all(len(c.scores) == 4 for c in result.columns)


def test_n_repeats_below_one_is_rejected(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_repeats must be >= 1"):
        permutation_importance(fitted, _constant, frame, n_repeats=0)


def test_negative_importance_survives_unclamped(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    """Shuffling an ignored column can help by luck; the sign is the signal."""
    rng = np.random.default_rng(0)

    def noisy(x: np.ndarray) -> np.ndarray:
        return x[:, 0, 0] + rng.normal(scale=5.0, size=x.shape[0])

    result = permutation_importance(fitted, noisy, frame, n_repeats=4)
    every_score = [s for c in result.columns for s in c.scores]
    assert any(s < 0 for s in every_score), "expected at least one negative drop"


def test_a_categorical_column_reports_once_not_once_per_level(
    frame: pd.DataFrame,
) -> None:
    """FR-009: aggregation is structural — the raw column is shuffled before one-hot."""
    cfg = ColumnConfig(
        input_columns=("feature_01", "feature_02", "comp_active"),
        output_column="target",
        categorical_columns=("comp_active",),
    )
    pre = Preprocessor(cfg).fit(frame)
    # One-hot expanded the categorical, so X is wider than the input list.
    assert pre.n_features > len(cfg.input_columns)

    result = permutation_importance(pre, _constant, frame, n_repeats=1)

    reported = [c.column for c in result.columns]
    assert reported.count("comp_active") == 1
    assert sorted(reported) == sorted(cfg.input_columns)


def test_every_input_column_is_reported(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    result = permutation_importance(fitted, _constant, frame, n_repeats=1)
    assert sorted(c.column for c in result.columns) == sorted(fitted.schema.inputs)


def test_a_frame_without_the_target_is_rejected(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="permutation importance needs a target"):
        permutation_importance(fitted, _constant, frame.drop(columns=["target"]), n_repeats=1)


def test_progress_fires_once_per_column_repeat(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    calls: list[tuple[int, int]] = []

    def record(completed: int, total: int) -> None:
        calls.append((completed, total))

    result = permutation_importance(fitted, _constant, frame, n_repeats=3, progress=record)

    expected = len(fitted.schema.inputs) * 3
    assert len(calls) == expected
    assert calls[0] == (1, expected)
    assert calls[-1] == (expected, expected)
    assert [c for c, _ in calls] == list(range(1, expected + 1))
    assert result.columns


def test_the_caller_frame_is_never_mutated(fitted: Preprocessor, frame: pd.DataFrame) -> None:
    before = frame.copy()
    permutation_importance(fitted, _constant, frame, n_repeats=2)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.slow
@pytest.mark.skipif(
    not (EXAMPLE_SESSION / "manifest.json").is_file(),
    reason="committed reference session is absent",
)
def test_end_to_end_against_the_committed_session() -> None:
    """The real Keras path, once: a loaded model scoring a real dataset."""
    from mlpp.data import read_table_auto
    from mlpp.predict import load_model, make_scorer
    from mlpp.predict_cli import columns_from
    from mlpp.session import load_session

    session = load_session(EXAMPLE_SESSION, columns_from(EXAMPLE_SESSION, strict_schema=True))
    scorer = make_scorer(load_model(session))
    dataset = read_table_auto(REPO_ROOT / "TEST" / "sample_test_A.csv")

    result = permutation_importance(session.preprocessor, scorer, dataset, n_repeats=2)

    assert len(result.columns) == len(session.preprocessor.schema.inputs)
    assert -1.0 <= result.baseline_r2 <= 1.0
    assert all(len(c.scores) == 2 for c in result.columns)

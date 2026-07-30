# Production Inference Seam — Plan Brief

> Full plan: `context/changes/s-04-production-inference-seam/plan.md`
> Frame brief: `context/changes/s-04-production-inference-seam/frame.md`

## What & Why

s-03 delivered a versioned artifact contract that nothing in production reads — the
engine can train and persist a session but cannot score a new CSV. This plan closes
the reader-side seam: a scoring path plus an `mlpp-predict` CLI, giving
`load_session()` its first production caller and closing `README.md`'s roadmap item #1.

## Starting Point

The gap is one function wide. `load_session()` (`session.py:310`) restores the
manifest and a fitted `Preprocessor`; `transform` already returns `y=None` for
target-less inference input; `inverse_target` already un-scales predictions to
engineering units. All three are TensorFlow-free and already exist. Even the scoring
expression is written once already, at `pipeline.py:200`, inside training's own
evaluation loop. What's missing is loading the model and calling `predict` — plus
two frictions: `DataConfig` demands directory paths a reader has no business
supplying, and unseen categorical levels are silently zero-encoded.

## Desired End State

`uv run mlpp-predict --session OUTPUTS/example --input TEST/some.csv --output preds.csv`
loads a session, validates its contract against disk, scores the CSV, and writes
predictions in engineering units — without importing the training pipeline. Unseen
categorical levels are reported rather than silently absorbed. The fast test suite
stays TensorFlow-free and ~1s.

## Key Decisions Made

| Decision | Choice | Why (1 sentence) | Source |
| --- | --- | --- | --- |
| Which epic next | Inference seam, then architectures, then dashboard | `load_session()` had zero production callers, so the contract had no reader. | Frame |
| Config seam | Split `ColumnConfig` out of `DataConfig` | Verified that `preprocess` and `session` read no directory field, so the split is mechanical and removes the need for fake paths. | Plan |
| Packaging | Accept TF; drop the "lightweight" claim | Scoring needs a Keras model, so it needs TensorFlow — engineering around that buys little and the honest doc costs nothing. | Plan |
| Model-kind in manifest | Defer; document the future bump | Only one framework exists today, and s-03's regenerate-don't-migrate rule makes a later `SCHEMA_VERSION` bump cheap. | Plan |
| Prediction output | Caller-named path, outside the session | Keeps session directories immutable and reproducible; scoring never mutates a training artifact. | Plan |
| Unseen categories | Detect and report, don't fail | `handle_unknown="ignore"` (`preprocess.py:98`) makes this a silent wrong answer at inference; a warning keeps batch runs alive. | Plan |
| HTTP seam | Out of scope → s-05 | A new deployable needs its own decisions; the reader gap is already closed by phase 3. | Plan |

## Scope

**In scope:**
- `ColumnConfig` extraction from `DataConfig`; narrowing `preprocess` and `session` to it
- New `predict.py` — the only new TensorFlow-owning module — with load/score split
- Fitted-categorical-level accessor on `Preprocessor` for unseen-level detection
- `mlpp-predict` CLI + entry point registration
- Tests for all three (including the first-ever CLI tests)
- Doc truth-up: README inference section + roadmap #1, and the stale `artifacts` module reference in `CLAUDE.md` / `__init__.py:4`

**Out of scope:**
- FastAPI / `apps/api` (→ s-05)
- Optional-dependency extras or any packaging split
- `model_kind` field, `SCHEMA_VERSION` bump
- Writing predictions into the session directory
- Metrics on predictions; batching or streaming for large inputs
- Refactoring how `best_model.keras` is written (`ModelCheckpoint` vs `register` split — noted, not touched)

## Architecture / Approach

```
ColumnConfig ──┐
               ├─> load_session() ──> LoadedSession ──┐
session dir ───┘   (TF-free)          manifest+pre    ├─> score_frame() ──> predictions
                                                      │    (TF-owning)      (engineering units)
input CSV ────────────────────────────────────────────┘
```

`predict.py` is the single new module permitted to import TensorFlow, keeping
`config`, `data`, `preprocess`, `metrics` and `session` cheap to import — the
property that keeps `pytest -m 'not slow'` at ~1s. It splits `load_model` from
`score_frame` so a future service pays the TF import once and scores many times.
The model filename is always resolved via `manifest.filenames_for(ROLE_BEST_MODEL)`,
never spelled outside `session.py`.

## Phases at a Glance

| Phase | What it delivers | Key risk |
| --- | --- | --- |
| 1. Extract `ColumnConfig` | Column contract separated from directory paths; `load_session` takes only what it uses | ~44 `DataConfig` references across 4 test modules; risk of leaving two live spellings of the same fields |
| 2. Scoring core | `predict.py`: load model, score frame, report unseen levels — first real consumer of `load_session()` | A TF import leaking into a TF-free layer would silently kill fast-suite latency |
| 3. `mlpp-predict` CLI + docs | Second entry point, first CLI tests, honest docs | Scope creep toward the deferred API; over-claiming lightness in the README |

**Prerequisites:** none — s-03 is landed, archived and pushed; suite is green (119 tests), mypy and ruff clean.
**Estimated effort:** ~2-3 sessions across 3 phases; phase 1 is the largest diff, phase 2 the highest value.

## Open Risks & Assumptions

- **Phase 1 is a wide but shallow refactor.** The `DataConfig` → `ColumnConfig` change touches four test modules; mechanical, but the diff is large enough to hide a mistake. The guard is that training behaviour must stay byte-identical and the committed reference run must keep loading.
- **`best_model.keras` is written by a Keras callback, not the session owner** (`training.py:51-52` vs `pipeline.py:118`). This plan reads the registered name and does not disturb the split — but it is the one place the single-owner invariant holds by convention rather than construction, and it will matter if a checkpoint is ever not written.
- **Unseen-level detection depends on fitted encoder state** surviving the joblib round trip. Assumed sound because `load_session` already restores and validates it, but it is an assumption this plan is the first to lean on.
- **Deferring `model_kind`** means Option B (LightGBM / foundation models) inherits a `SCHEMA_VERSION` bump. Accepted deliberately; a stray non-`.keras` model file would fail confusingly until then.

## Success Criteria (Summary)

- Someone with the repo and a trained session can score a new CSV with one command and get predictions in real engineering units.
- Unseen categorical levels surface as a warning instead of a silently wrong number.
- The fast suite stays TensorFlow-free and ~1s; full suite, mypy and ruff stay clean.

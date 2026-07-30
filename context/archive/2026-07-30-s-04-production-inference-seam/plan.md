# Production Inference Seam Implementation Plan

## Overview

Give the s-03 manifest contract its first production reader. Today the engine can
train a session and persist a validated artifact directory, but nothing can score
a new CSV against it — `load_session()` has zero production callers. This plan
adds the missing scoring path and exposes it as `mlpp-predict`.

## Current State Analysis

The reader-side seam is one function short of working. Three of the four steps a
predict path needs already exist and are TensorFlow-free:

| Step | Where it lives | Status |
| --- | --- | --- |
| Restore manifest + fitted `Preprocessor` | `session.py:310` `load_session` | exists, no production caller |
| Align + transform target-less input | `preprocess.py:164-165` (`transform` returns `y=None`) | exists |
| Un-scale predictions to engineering units | `preprocess.py:195-199` `inverse_target` | exists |
| **Load the model and call `predict`** | — | **missing** |

The scoring expression itself is already written once, inside training's own
evaluation loop: `pipeline.py:200` does
`y_pred = pre.inverse_target(model.predict(x_test, verbose=0))`. The new module
mirrors that line; the genuinely new work is model loading, unseen-category
detection, and the entry point around it.

Two frictions block a clean predict path:

1. **`DataConfig` conflates location with contract.** `load_session(session_dir, cfg)`
   demands a full `DataConfig` whose `train_dir`/`test_dir`/`output_dir` are
   meaningless at predict time, while its `input_columns`/`strict_schema` are
   load-bearing (`preprocess.py:170` `align_columns`). Verified: neither
   `preprocess.py` nor `session.py` reads *any* directory field — they only ever
   use the column half. The split is therefore mechanical, not speculative.
2. **`best_model.keras` is written outside the session owner.** `training.py:51-52`
   hands `str(out_dir / BEST_MODEL_FILE)` to a Keras `ModelCheckpoint`, while
   `pipeline.py:118` separately calls `writer.register(ROLE_BEST_MODEL, ...)`.
   Registration and writing are split across two modules. The reader only needs
   the registered name, so this plan does not disturb it — but it is the one place
   the single-owner invariant is honoured by convention rather than by construction.

## Desired End State

`uv run mlpp-predict --session OUTPUTS/example --input TEST/some.csv --output preds.csv`
loads a session, validates its contract, scores the CSV, and writes predictions in
engineering units — without importing the training pipeline. `load_session()` has a
real caller, and `README.md:149`'s roadmap item #1 is checked.

Verify by: running the command against the committed `OUTPUTS/example/` and a file
from `TEST/`, confirming a predictions CSV with one row per input row; and by
confirming the fast suite still imports no TensorFlow.

### Key Discoveries:

- `load_session` has **zero production callers** — `session.py:310`, referenced only by `tests/test_session.py` and `tests/test_pipeline.py`.
- `session.py:316-317` reserves scoring for a predict CLI by name: *"scoring belongs to the predict CLI, not to the contract."*
- Reference scoring line already exists at `pipeline.py:200`.
- `preprocess.py` and `session.py` use **only** the column fields of `DataConfig` — no directory field appears in either (verified by grep). ColumnConfig extraction is therefore non-invasive.
- `OneHotEncoder(handle_unknown="ignore")` at `preprocess.py:98` silently zero-encodes unseen categorical levels. Harmless at fit time, a silent wrong answer at inference time.
- `align_columns` (`preprocess.py:64-78`) already raises `SchemaError` for missing input columns under `strict_schema` (default `True`), else fills `0.0`.
- The manifest is already framework-agnostic: `ArtifactEntry` is `{role, filename}` (`session.py:120-126`), so no schema change is needed for this plan.
- **Stale docs:** `src/mlpp/artifacts.py` does not exist, yet `CLAUDE.md` and the `__init__.py:4` docstring both name `artifacts` as a TF-free module. s-03 folded it into `session.py` without updating either.
- There is no `tests/test_cli.py` — the CLI is currently untested.

## What We're NOT Doing

- **No FastAPI / `apps/api`.** Deferred to s-05; a new deployable needs its own decisions.
- **No packaging or extras split.** TensorFlow stays an unconditional runtime dependency (`pyproject.toml:9`). The "lightweight inference" claim is dropped from the docs rather than engineered for — scoring needs a Keras model, so it needs TF.
- **No `model_kind` field and no `SCHEMA_VERSION` bump.** The loader infers from the `.keras` extension; Option B (LightGBM / foundation models) will pay for the bump, which s-03's regenerate-don't-migrate rule makes cheap.
- **No predictions in the session directory.** Session dirs stay immutable and reproducible; no `ROLE_PREDICTIONS`.
- **No changes to how `best_model.keras` is written.** The `ModelCheckpoint`/`register` split is noted, not refactored.
- **No metrics on predictions.** Scoring an unlabelled CSV produces predictions only; comparing against truth stays training's job.
- **No batching / streaming for large inputs.** Single-frame scoring only.

## Implementation Approach

Three phases, strictly ordered: the config seam must land before the scoring core
can have an honest signature, and the CLI needs the core to exist.

Phase 1 is a pure refactor with no behaviour change — it extracts the column
contract from `DataConfig` into a `ColumnConfig` that `DataConfig` composes, so
`preprocess` and `session` depend only on what they actually use. Phase 2 adds the
one missing capability in a new TensorFlow-owning module, keeping every existing
TF-free layer TF-free. Phase 3 exposes it and trues up the docs.

## Critical Implementation Details

**Ordering.** Phase 1 changes `load_session`'s signature, which Phase 2 consumes.
Doing Phase 2 first would mean writing the scoring core against a signature that is
about to change.

**Import hygiene is the load-bearing constraint.** The new scoring module is the
*only* new file allowed to import TensorFlow/Keras. If a TF import leaks into
`session`, `preprocess`, `config`, `data` or `metrics`, the fast suite
(`pytest -m 'not slow'`, ~1s) stops being fast — that separation is the point.
The new module's tests must therefore be marked `@pytest.mark.slow`.

**Unseen-category detection reads fitted state, not config.** The comparison is
between the input frame's categorical values and the *fitted* `OneHotEncoder`'s
`categories_`, reached through the restored `Preprocessor` — not against
`cfg.categorical_columns`, which only names which columns are categorical, not
which levels were seen.

## Phase 1: Extract ColumnConfig from DataConfig

### Overview

Split the column contract out of `DataConfig` so the predict path can supply it
without inventing meaningless directory paths. Pure refactor — no behaviour change,
no manifest change.

### Changes Required:

#### 1. Config dataclasses

**File**: `apps/backend/src/mlpp/config.py`

**Intent**: Introduce a `ColumnConfig` holding the column contract, and have
`DataConfig` compose it so training keeps a single object. Preserve the existing
`DataConfig` field access used by `cli.py` and tests wherever it is cheap to do so.

**Contract**: New frozen slotted dataclass carrying the four column fields that
`preprocess` actually reads:

```python
@dataclass(frozen=True, slots=True)
class ColumnConfig:
    input_columns: tuple[str, ...] = DEFAULT_INPUT_COLUMNS
    output_column: str = DEFAULT_OUTPUT_COLUMN
    categorical_columns: tuple[str, ...] = ()
    strict_schema: bool = True
```

`DataConfig` keeps `train_dir`, `test_dir`, `output_dir`, `test_files` and gains a
`columns: ColumnConfig` field. Decide and apply one consistent access story for the
existing `cfg.input_columns` call sites — either forwarding properties on
`DataConfig` or updated call sites — and do not leave both spellings live.

#### 2. Preprocessing takes the column contract only

**File**: `apps/backend/src/mlpp/preprocess.py`

**Intent**: Narrow `resolve_schema`, `align_columns`, `Preprocessor.__init__` and
`Preprocessor.restore` to accept `ColumnConfig` instead of `DataConfig`, since they
provably use nothing else.

**Contract**: The four signatures at `preprocess.py:49`, `:64`, `:104` and the
`restore` classmethod change their `cfg` parameter type from `DataConfig` to
`ColumnConfig`. No logic changes.

#### 3. Session loading takes the column contract only

**File**: `apps/backend/src/mlpp/session.py`

**Intent**: Change `load_session` to accept `ColumnConfig`, removing the demand for
directory paths a reader has no business supplying.

**Contract**: `load_session(session_dir: Path, cfg: ColumnConfig, *, use_onehot: bool = True) -> LoadedSession`.
The `TYPE_CHECKING`/import of `DataConfig` in this module goes away.

#### 4. Call sites

**File**: `apps/backend/src/mlpp/cli.py`, `apps/backend/src/mlpp/pipeline.py`

**Intent**: Build and thread the composed config so training behaviour is byte-identical.

**Contract**: `config_from_args` (`cli.py:64`) constructs `ColumnConfig` inside the
`DataConfig` it already builds; `pipeline.py` passes the column half where a
`Preprocessor` is constructed or restored.

#### 5. Tests

**File**: `apps/backend/tests/test_config.py`, `test_preprocess.py`, `test_session.py`, `conftest.py`

**Intent**: Update fixtures and assertions to the new shape, and add coverage that
`ColumnConfig` alone is sufficient to restore a session.

**Contract**: `conftest.py` gains (or adapts) a `ColumnConfig` fixture; the ~44
existing `DataConfig` references across these four files are updated. Add one test
asserting `load_session` works with a `ColumnConfig` built from nothing but the
manifest's own feature contract — the property Phase 2 depends on.

### Success Criteria:

#### Automated Verification:

- Fast suite passes: `cd apps/backend && uv run pytest -m 'not slow'`
- Full suite passes: `cd apps/backend && uv run pytest`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- No TensorFlow import reaches the TF-free layers: `session`, `preprocess`, `config`, `data`, `metrics`
- A real training run still completes: `uv run mlpp-train --epochs 1 --no-plots --quiet`
- The committed reference run still loads (no manifest change): `load_session` against `OUTPUTS/example/`

#### Manual Verification:

- Only one spelling of the column fields is live — no lingering duplicate access path left behind
- A reader of `config.py` can tell at a glance which fields are "where data lives" vs "what the columns are"

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding.

---

## Phase 2: Scoring core

### Overview

Add the one missing capability: load a session's model and score a frame, in a new
module that owns the TensorFlow dependency. This is the first production consumer of
`load_session()`.

### Changes Required:

#### 1. New scoring module

**File**: `apps/backend/src/mlpp/predict.py` (new)

**Intent**: Load the registered best model from a session directory and score a
DataFrame, returning predictions in engineering units. Report unseen categorical
levels rather than letting them pass silently.

**Contract**: Two public entry points — one to load, one to score — so a caller
(CLI now, service later) can load once and score many times:

```python
def load_model(session: LoadedSession) -> keras.Model: ...

def score_frame(
    session: LoadedSession, model: keras.Model, df: pd.DataFrame
) -> PredictionResult: ...
```

`load_model` resolves the filename via `session.manifest.filenames_for(ROLE_BEST_MODEL)`
— never by spelling `best_model.keras` — and raises `ArtifactError` if the role is
absent or holds more than one entry. `score_frame` runs
`transform` → `model.predict` → `inverse_target`, mirroring `pipeline.py:200`.
`PredictionResult` is a frozen dataclass carrying at minimum the predictions array
and the unseen-level report (column → set of unrecognised values, plus affected row
count), so the caller decides how loudly to surface it.

#### 2. Unseen-category detection

**File**: `apps/backend/src/mlpp/preprocess.py`

**Intent**: Expose the fitted categorical levels so a caller can compare input
values against what the encoder actually saw, closing the `handle_unknown="ignore"`
blind spot.

**Contract**: A read-only accessor on `Preprocessor` returning the fitted levels per
categorical column, sourced from the restored `OneHotEncoder.categories_`. Raises
`NotFittedError` when unfitted, consistent with the rest of the class. Stays
TensorFlow-free.

#### 3. Tests

**File**: `apps/backend/tests/test_predict.py` (new)

**Intent**: Cover the round trip end to end and the failure modes that matter.

**Contract**: All tests marked `@pytest.mark.slow` (the module imports Keras). Cover:
score a frame against a freshly trained session and get one prediction per row;
predictions come back in engineering units, not scaled space; unseen categorical
levels are reported; a missing model role raises `ArtifactError`; a target-less
input frame scores fine (the `y=None` path); and a session whose `schema_version`
differs still raises `SchemaVersionError` through this path.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- New tests pass: `uv run pytest tests/test_predict.py -v`
- Fast suite is still TensorFlow-free and still fast: `uv run pytest -m 'not slow'`
- `predict.py` is the only new module importing TensorFlow/Keras
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- Scoring the committed `OUTPUTS/example/` against a `TEST/*.csv` produces one prediction per input row

#### Manual Verification:

- Predicted values are in a plausible range for the target column — not centred on zero, which would indicate `inverse_target` was skipped
- The unseen-level report reads clearly enough for an operator to act on it
- No filename literal for the model appears outside `session.py`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding.

---

## Phase 3: mlpp-predict CLI and doc truth-up

### Overview

Expose the scoring core as the project's second entry point, and correct the docs
that currently describe a module that no longer exists and a roadmap item that is
now done.

### Changes Required:

#### 1. Predict CLI

**File**: `apps/backend/src/mlpp/predict_cli.py` (new)

**Intent**: A thin argument-parsing layer over `predict.py` that loads a session,
scores an input CSV, and writes predictions to a caller-named path. Predictions are
never written into the session directory.

**Contract**: `argparse`-based, mirroring `cli.py:25`'s `build_parser` /
`config_from_args` / `main` shape and returning an exit code from `main`. Required
`--session DIR`, `--input CSV`, `--output CSV`; column overrides mirroring the
train CLI's flags so a non-default schema can be scored. Reads the column contract
from the manifest by default. Surfaces the unseen-level report on stderr as a
warning and still exits `0`; genuine failures raise from the `MlppError` hierarchy
and map to a non-zero exit.

#### 2. Entry point registration

**File**: `apps/backend/pyproject.toml`

**Intent**: Register the second console script.

**Contract**: `mlpp-predict = "mlpp.predict_cli:main"` alongside the existing
`mlpp-train` at `pyproject.toml:19`.

#### 3. CLI tests

**File**: `apps/backend/tests/test_predict_cli.py` (new)

**Intent**: Cover argument handling and exit codes — the CLI currently has no test
module at all.

**Contract**: Marked `@pytest.mark.slow`. Cover: end-to-end invocation writes the
expected output file; a missing `--session` directory exits non-zero with an
actionable message; a schema-mismatched input exits non-zero; unseen levels warn but
exit `0`.

#### 4. Documentation

**File**: `README.md`, `CLAUDE.md`, `apps/backend/src/mlpp/__init__.py`

**Intent**: Document the new command, check off the roadmap item it closes, and fix
the stale `artifacts` module reference in both the project instructions and the
package docstring.

**Contract**: `README.md` gains an inference usage section and checks
`- [x] mlpp-predict CLI` at line 149. `CLAUDE.md`'s TF-free module list and the
`__init__.py:4` docstring drop `artifacts` (folded into `session.py` by s-03) and
name the real TF-free set. `CLAUDE.md`'s build/run section gains the
`mlpp-predict` invocation. State plainly that inference requires TensorFlow — do not
describe the install as lightweight.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- New CLI tests pass: `uv run pytest tests/test_predict_cli.py -v`
- The entry point resolves: `uv run mlpp-predict --help`
- End-to-end scoring works: `uv run mlpp-predict --session ../../OUTPUTS/example --input <a TEST csv> --output /tmp/preds.csv`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- No stale `artifacts` module reference remains: `grep -rn "artifacts" CLAUDE.md src/mlpp/__init__.py` returns nothing describing it as a module
- CI passes on the branch (both Python versions plus notebook-drift)

#### Manual Verification:

- `--help` output is clear enough to use the command without reading the source
- The output CSV opens cleanly and its columns are self-explanatory
- `README.md`'s inference section is accurate about the TensorFlow requirement — no lightweight claim survives
- A reader following `README.md` alone can score the committed example run

**Implementation Note**: After completing this phase and all automated verification passes, pause for manual confirmation from the human.

---

## Testing Strategy

### Unit Tests:

- `ColumnConfig` is sufficient on its own to restore a session (Phase 1) — the property the whole plan rests on
- Fitted-level accessor raises `NotFittedError` when unfitted
- `load_model` raises `ArtifactError` on a missing or ambiguous model role

### Integration Tests:

- Train a session, then score a frame through `load_session` → `load_model` → `score_frame` and assert one prediction per row in engineering units
- Score the committed `OUTPUTS/example/` — guards the reference run against silent rot, the exact failure s-03 fixed
- CLI end to end: session dir + input CSV → output CSV, correct exit codes

### Manual Testing Steps:

1. `uv run mlpp-train --epochs 1 --no-plots --quiet`, note the new session directory
2. `uv run mlpp-predict --session <that dir> --input <a TEST csv> --output /tmp/preds.csv`
3. Open `/tmp/preds.csv` and sanity-check the values are in the target's real range
4. Score against `OUTPUTS/example/` and confirm identical behaviour on the committed run
5. Feed a CSV with an unseen categorical level; confirm a warning and exit `0`
6. Feed a CSV missing a required input column; confirm a clear `SchemaError` and non-zero exit

## Performance Considerations

Scoring is a single forward pass over a small matrix; s-03 measured the data path at
0.5% of runtime on a 76 KB matrix, so no optimisation is warranted. The one
performance property worth protecting is **fast-suite latency**: keeping TF out of
every existing module is why `pytest -m 'not slow'` runs in ~1s, and the new module
must not erode it. Process startup for `mlpp-predict` is dominated by importing
TensorFlow (seconds) — acceptable for batch scoring, and the reason the load/score
split exists so a future service pays it once.

## Migration Notes

No manifest change and no `SCHEMA_VERSION` bump, so the committed `OUTPUTS/example/`
keeps loading and no session needs regenerating. `load_session`'s signature change
is internal — its only callers are two test modules. `DataConfig` is a public-ish
dataclass; the composition change is source-breaking for any external caller
constructing it positionally, which within this repo is only tests and `cli.py`.

## References

- Frame brief: `context/changes/s-04-production-inference-seam/frame.md`
- Reference scoring line: `apps/backend/src/mlpp/pipeline.py:200`
- Contract reader: `apps/backend/src/mlpp/session.py:310` (`load_session`), `:120-148` (manifest shape)
- Inference-ready preprocessing: `apps/backend/src/mlpp/preprocess.py:164-165` (`transform`), `:195-199` (`inverse_target`), `:98` (`handle_unknown="ignore"`), `:64-78` (`align_columns`)
- CLI pattern to mirror: `apps/backend/src/mlpp/cli.py:25-90`
- Checkpoint writer: `apps/backend/src/mlpp/training.py:51-52`
- Roadmap item closed: `README.md:149`
- Prior change: `context/archive/2026-07-29-s-03-algorithm-and-architecture-deep-optimization/plan.md`

## Progress

> Convention: `- [ ]` pending, `- [x]` done. Append ` — <commit sha>` when a step lands. Do not rename step titles. See `references/progress-format.md`.

### Phase 1: Extract ColumnConfig from DataConfig

#### Automated

- [x] 1.1 Fast suite passes: `cd apps/backend && uv run pytest -m 'not slow'` — 4c2411a
- [x] 1.2 Full suite passes: `cd apps/backend && uv run pytest` — 4c2411a
- [x] 1.3 Type checking passes: `uv run mypy src/mlpp` — 4c2411a
- [x] 1.4 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — 4c2411a
- [x] 1.5 No TensorFlow import reaches `session`, `preprocess`, `config`, `data`, `metrics` — 4c2411a
- [x] 1.6 A real training run still completes: `uv run mlpp-train --epochs 1 --no-plots --quiet` — 4c2411a
- [x] 1.7 The committed reference run still loads via `load_session` against `OUTPUTS/example/` — 4c2411a

#### Manual

- [x] 1.8 Only one spelling of the column fields is live — no duplicate access path left behind — 4c2411a
- [x] 1.9 `config.py` makes the "where data lives" vs "what the columns are" split obvious at a glance — 4c2411a

### Phase 2: Scoring core

#### Automated

- [x] 2.1 Full suite passes: `cd apps/backend && uv run pytest` — fa62bd7
- [x] 2.2 New tests pass: `uv run pytest tests/test_predict.py -v` — fa62bd7
- [x] 2.3 Fast suite is still TensorFlow-free and still fast: `uv run pytest -m 'not slow'` — fa62bd7
- [x] 2.4 `predict.py` is the only new module importing TensorFlow/Keras — fa62bd7
- [x] 2.5 Type checking passes: `uv run mypy src/mlpp` — fa62bd7
- [x] 2.6 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — fa62bd7
- [x] 2.7 Scoring `OUTPUTS/example/` against a `TEST/*.csv` produces one prediction per input row — fa62bd7

#### Manual

- [x] 2.8 Predicted values fall in a plausible range for the target column (not centred on zero) — fa62bd7
- [x] 2.9 The unseen-level report reads clearly enough for an operator to act on it — fa62bd7
- [x] 2.10 No filename literal for the model appears outside `session.py` — fa62bd7

### Phase 3: mlpp-predict CLI and doc truth-up

#### Automated

- [x] 3.1 Full suite passes: `cd apps/backend && uv run pytest` — 0ff9a9c
- [x] 3.2 New CLI tests pass: `uv run pytest tests/test_predict_cli.py -v` — 0ff9a9c
- [x] 3.3 The entry point resolves: `uv run mlpp-predict --help` — 0ff9a9c
- [x] 3.4 End-to-end scoring works against `OUTPUTS/example/` with `--output` — 0ff9a9c
- [x] 3.5 Type checking passes: `uv run mypy src/mlpp` — 0ff9a9c
- [x] 3.6 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — 0ff9a9c
- [x] 3.7 No stale `artifacts` module reference remains in `CLAUDE.md` or `src/mlpp/__init__.py` — 0ff9a9c
- [x] 3.8 CI passes on the branch (both Python versions plus notebook-drift) — 0ff9a9c

#### Manual

- [x] 3.9 `--help` output is usable without reading the source — 0ff9a9c
- [x] 3.10 The output CSV opens cleanly and its columns are self-explanatory — 0ff9a9c
- [x] 3.11 `README.md`'s inference section is accurate about the TensorFlow requirement — 0ff9a9c
- [x] 3.12 A reader following `README.md` alone can score the committed example run — 0ff9a9c

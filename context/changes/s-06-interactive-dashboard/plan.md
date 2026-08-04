# Interactive Dashboard Implementation Plan

## Overview

Add a local browser-based dashboard over `mlpp` that delivers four surfaces to the model author:
session introspection, global permutation feature importance, single-row inference, and batch
CSV/Parquet inference. Three of the four are presentation layers over capability that already
exists. The fourth — feature importance — is new domain logic and is built as a TensorFlow-free
library module, so it stays testable in the fast suite and usable outside the interface.

## Current State Analysis

The package is a `mlpp` uv project under `apps/backend/` with two console entrypoints and no
interface of any kind. Inspecting a trained session means opening a directory of CSVs, PNGs and
standalone HTML reports by hand; scoring means composing a three-argument `mlpp-predict` call
against a CSV written to disk first.

Constraints discovered during research, all load-bearing:

- **The TensorFlow-free / TensorFlow-owning module split is real and verified.** Running the entire
  98-test fast suite leaves neither `tensorflow` nor `keras` in `sys.modules`. `config`, `data`,
  `preprocess`, `metrics`, `session` and `errors` import no TensorFlow; `model`, `training`,
  `plots`, `pipeline` and `predict` do. PRD guardrail FR-019 makes preserving this non-negotiable.
- **`session.py` is the sole owner of every filename in a session directory** (`session.py:36-53`).
  The module docstring records that three modules once each persisted overlapping copies of the
  feature contract, drifted, and silently broke the committed reference run. FR-020 preserves this.
- **`read_csv_auto` has three call sites** — `predict_cli.py:106` (scoring) and `pipeline.py:90`
  and `pipeline.py:193` (training and evaluation). Only the first is in scope.
- **Two blockers sit outside the code.** `CLAUDE.md` and `.cursorrules:8` both instruct that no
  frontend may exist. And a `joblib`/`numpy` deprecation fires on `session.read_fitted_state()` —
  the path every panel will depend on.

## Desired End State

From `apps/backend/`, one command starts a local dashboard. The author selects any session
directory under `OUTPUTS/` and sees its architecture, feature contract, training history and test
metrics; computes a ranked feature-importance chart against a dataset they choose; scores a
hand-entered row with an out-of-range warning; and scores a whole CSV or Parquet file with a
download. An invalid session produces a message naming the problem rather than a traceback.

Verify by: `uv run pytest` green (145 existing + new tests), `uv run pytest -m 'not slow'` still
TensorFlow-free and ~1.5s, `uv run mypy src/mlpp` clean, `ruff check` / `ruff format --check`
clean, and the manual walkthrough in Testing Strategy below.

### Key Discoveries:

- **Permuting the raw DataFrame column rather than `X` makes FR-009 free.** `preprocess.py:199-200`
  one-hot expands *after* `align_columns`, so shuffling a source column before `transform`
  reshuffles all of that column's one-hot positions together by construction. Permuting `X`
  directly would require regrouping positions via `feature_names` — more code, more ways to be
  wrong.
- **The permutation loop needs no TensorFlow.** `metrics.regression_metrics` (`metrics.py:23`)
  computes R² and is TF-free; `Preprocessor.transform` (`preprocess.py:187`) is TF-free. Only
  `model.predict` needs TF. Parameterising the loop over a scorer callable puts the whole module in
  the fast suite.
- **`transform` already returns `y`** (`preprocess.py:187`), so the baseline score needs no
  separate path.
- **`_fill_missing` uses the column mean for numeric NaNs** (`preprocess.py:87-89`). Permuting a
  column does not change its mean, so fill behaviour is stable across permutations — the baseline
  and permuted runs stay comparable.
- **`PredictionResult.unseen`** (`predict.py:34-51`) already reports unrecognised categorical
  levels with a ready-made `describe_unseen()`; FR-016 surfaces what exists rather than computing
  anything new.
- **`Preprocessor` exposes `_scaler` as fitted state** via `fitted_state` (`preprocess.py:235`).
  `StandardScaler` carries `mean_` and `scale_`, which is what FR-022's range check reads.

## What We're NOT Doing

- No training from the dashboard. No run launching, no checkpointing UI.
- No SHAP, captum, or gradient attribution. No per-row local explanations — global importance only.
- No hosting, container image, authentication, or multi-user access.
- No session mutation: no delete, repair, rename, or regenerate. The three incomplete session
  directories under `OUTPUTS/` are reported on, never fixed.
- No new artifact role and no `SCHEMA_VERSION` bump.
- No change to `read_csv_auto` itself, and no change to the training path's use of it.
- No fix for the underlying `joblib`/`numpy` deprecation — this plan arms a tripwire only; the
  version decision is its own change.
- No `.editorconfig`, `py.typed`, or Dependabot work from the health check. Out of scope here.

## Implementation Approach

Five phases, ordered to front-load risk. Phase 1 clears the two prerequisites and declares
dependencies without writing dashboard code, so every later phase runs against instructions that
permit and describe the work. Phase 2 adds the Parquet reader as a sibling of `read_csv_auto`,
leaving the training path untouched by construction. Phase 3 builds feature importance as a pure
library module *before any interface exists*, which forces the FR-011 boundary rather than
retrofitting it. Phases 4 and 5 add the interface, which by then is assembly rather than design.

The dashboard package sits at `apps/backend/src/mlpp/dashboard/`, on the TensorFlow-owning side of
the module split. It renders; it does not compute.

## Critical Implementation Details

**State sequencing in the permutation loop.** The baseline must be scored from the *unpermuted*
frame once, before any column is touched, and each permutation must start from a fresh copy of the
original frame rather than from the previously-permuted one. Permuting cumulatively — the obvious
loop if you mutate in place — measures the joint degradation of every column processed so far, not
the marginal contribution of each, and produces a monotonically increasing curve that looks
plausible while being wrong.

**Negative importance is signal, not error.** A column the model ignores can score slightly
negative because shuffling it happened to help. Do not clamp to zero: the sign carries information
about whether a feature is genuinely unused, and FR-009's acceptance criteria require it displayed
as-is.

## Phase 1: Prerequisites & contract

### Overview

Clear both blockers and declare dependencies. No dashboard code lands in this phase. Its purpose is
that every subsequent phase runs against instruction files that permit and describe the work, with
a tripwire under the artifact-loading path.

### Changes Required:

#### 1. Project instructions

**File**: `CLAUDE.md`

**Intent**: Stop the file forbidding the work being planned, and supply the layout conventions
Streamlit does not. Without this, every agent session re-litigates whether the dashboard should
exist.

**Contract**: Four edits. In `## What this is`, replace the "Python only — there is no frontend and
no infra stack" sentence so it distinguishes a JavaScript frontend (still absent) from the Streamlit
dashboard (Python, inside the backend package). In `## Project boundaries` → Layout, add the
`apps/backend/src/mlpp/dashboard/` entry naming it TensorFlow-owning. In `## Core tech stack`, add
the Keras 3 API-surface rule and the Streamlit line. In `## Build & run`, add the dashboard run
command. Add a new `## Dashboard conventions` section — paste-ready text is in
`context/foundation/stack-assessment.md` under "Recommended Instruction File Additions".

**File**: `.cursorrules`

**Intent**: Line 8 asserts "no frontend/infra here", which contradicts the layout after this change.
The file opens with "Obey literally", so it will be followed as written.

**Contract**: Rewrite the `LAYOUT:` line to permit `mlpp.dashboard` as Python-in-backend while
still excluding a JavaScript frontend. Keep the file's dense, token-optimized style — it is a
different register from `CLAUDE.md` deliberately.

#### 2. Dependency declarations

**File**: `apps/backend/pyproject.toml`

**Intent**: Declare `pyarrow` in base dependencies so `data.py` — a core module — can honestly
guarantee the Parquet capability Phase 2 gives it. Keep the existing uncommitted `dashboard` group
and build on it.

**Contract**: Add `pyarrow` to `[project].dependencies` beside `pandas`. Leave the existing
`[dependency-groups].dashboard` entry (`streamlit>=1.40,<2`) as-is. `uv lock` afterwards; the
existing 295-line resolution already carries pyarrow transitively, so the delta should be a
promotion rather than a new download.

#### 3. Deprecation tripwire

**File**: `apps/backend/pyproject.toml`

**Intent**: The `joblib`/`numpy` deprecation on `read_fitted_state()` is currently invisible to the
suite — `filterwarnings = ["error::DeprecationWarning:mlpp.*"]` matches only warnings originating in
`mlpp` modules, and this one originates in `joblib`. Converting it to a failure means the future
removal arrives as a red test rather than a silent break that takes `OUTPUTS/example/` with it.

**Contract**: Add `"error::DeprecationWarning:joblib.*"` to `[tool.pytest.ini_options]`
`filterwarnings`. Expect this to fail immediately — 30 warnings currently fire from
`tests/test_session.py`. That failure is the point, so pair it with a narrowly-scoped ignore
carrying a comment naming the condition and pointing at the follow-up change, so the tripwire fires
on *new* joblib deprecations while the known one stays documented rather than silent.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- Fast suite still TensorFlow-free: `uv run pytest -m 'not slow'`
- Lockfile resolves and is not stale: `uv lock --check`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- `pyarrow` is importable from a base install: `uv run --no-group dashboard python -c "import pyarrow"`
- The joblib tripwire fires on an unignored deprecation (verify by temporarily removing the scoped ignore)

#### Manual Verification:

- No sentence in `CLAUDE.md` or `.cursorrules` now forbids the dashboard
- The `## Dashboard conventions` section describes the layout Phases 4–5 will actually build
- The scoped joblib ignore names the condition and points at the follow-up change

**Implementation Note**: After completing this phase and all automated verification passes, pause
for manual confirmation before proceeding.

---

## Phase 2: Parquet reader

### Overview

Give the package a Parquet read path as an additive sibling of `read_csv_auto`, and wire the
scoring entrypoint to it. The training path is untouched by construction.

### Changes Required:

#### 1. Suffix-dispatching reader

**File**: `apps/backend/src/mlpp/data.py`

**Intent**: Add a reader that dispatches on file extension so any scoring input can be CSV or
Parquet, without modifying the delimiter-sniffing CSV path that the training pipeline depends on.

**Contract**: New public function `read_table_auto(path: Path) -> pd.DataFrame`. Dispatches
`.parquet` / `.pq` to `pd.read_parquet`, everything else to the existing `read_csv_auto`. Raises
`DatasetError` — not a bare pandas exception — when a Parquet file is unreadable or yields zero
rows, matching `read_csv_auto`'s contract so callers handle one error type. Reuses the existing
`_require_rows` helper. `read_csv_auto` itself is not modified. The module stays TensorFlow-free.

#### 2. Scoring entrypoint

**File**: `apps/backend/src/mlpp/predict_cli.py`

**Intent**: `mlpp-predict` reads scoring input through the new dispatcher, so Parquet is a package
capability rather than an interface-only one. Additive: CSV behaviour is unchanged.

**Contract**: Swap the import and the call site at line 106 from `read_csv_auto` to
`read_table_auto`. The `--input` help text gains Parquet. No flag changes — FR-018 preserves the
CLI's surface, and suffix dispatch needs no new option.

#### 3. Tests

**File**: `apps/backend/tests/test_data.py`

**Intent**: Cover the dispatcher's routing and its error contract. Fast suite — no TensorFlow.

**Contract**: Round-trip a Parquet file; confirm `.csv` still routes to the sniffing reader with all
existing delimiters; confirm a corrupt Parquet raises `DatasetError` rather than a pyarrow
exception; confirm an empty Parquet raises `DatasetError`. Existing `read_csv_auto` tests stay
untouched — they are the regression guard for the training path.

**File**: `apps/backend/tests/test_predict_cli.py`

**Intent**: Prove the preserved-CLI claim rather than asserting it.

**Contract**: Add a Parquet-input case alongside the existing CSV cases, and confirm CSV scoring
produces identical output to before the swap. Module is already `pytestmark = pytest.mark.slow`.

### Success Criteria:

#### Automated Verification:

- New data tests pass: `uv run pytest tests/test_data.py -v`
- CLI tests pass: `uv run pytest tests/test_predict_cli.py -v`
- Full suite passes: `uv run pytest`
- Fast suite stays TensorFlow-free and ~1.5s: `uv run pytest -m 'not slow'`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- Training path untouched: `read_csv_auto` still has exactly two call sites in `pipeline.py`

#### Manual Verification:

- `uv run mlpp-predict --session ../../OUTPUTS/example --input <a .parquet> --output /tmp/p.csv` scores correctly
- The same command against the existing `TEST/sample_test_A.csv` produces unchanged output

**Implementation Note**: Pause for manual confirmation before proceeding.

---

## Phase 3: Feature importance

### Overview

The only genuinely new domain logic in this change. Built as a library module with no interface
dependency and no mandatory TensorFlow import, so it satisfies FR-011 and FR-019 together rather
than trading one against the other.

### Changes Required:

#### 1. The importance module

**File**: `apps/backend/src/mlpp/importance.py` (new)

**Intent**: Compute how much the model's accuracy degrades when each input column's values are
shuffled — the domain rule from the PRD. Parameterised over a scorer callable so the module itself
imports no TensorFlow and the loop is exercisable in the fast suite with a stub.

**Contract**: The module's signature is the contract other phases depend on, so it is pinned here:

```python
Scorer = Callable[[np.ndarray], np.ndarray]   # X (rows, n_features, 1) -> raw predictions

@dataclass(frozen=True, slots=True)
class ColumnImportance:
    column: str          # an *input* column, not a one-hot expansion
    mean_drop: float     # mean R² degradation; may be negative
    std_drop: float      # spread across repeats
    scores: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class ImportanceResult:
    baseline_r2: float
    columns: tuple[ColumnImportance, ...]   # sorted by mean_drop, descending
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
) -> ImportanceResult: ...
```

Algorithm: transform the frame once and score it to establish `baseline_r2` via
`metrics.regression_metrics`. Then for each column in `preprocessor.schema.inputs`, and for each
repeat, take a **fresh copy** of the original frame, shuffle that one column with a seeded
`np.random.default_rng(seed + repeat)`, re-run `transform`, re-score, and record
`baseline_r2 - permuted_r2`. Iterating over `schema.inputs` rather than `feature_names` is what
delivers FR-009 — a categorical column is shuffled as a unit before one-hot expansion, so
aggregation is structural rather than a post-hoc regroup.

Raises `SchemaError` when the frame carries no target column — importance is undefined without one.
`progress` is invoked as `(completed, total)` so a caller can render a bar without this module
importing any interface library.

#### 2. Keras-backed scorer

**File**: `apps/backend/src/mlpp/predict.py`

**Intent**: Supply the one TensorFlow-dependent piece — a `Scorer` closure over a loaded model — so
`importance.py` never imports Keras itself.

**Contract**: New function `make_scorer(model: keras.Model) -> Scorer` returning a callable that
runs `model.predict(x, verbose=0)`. This module already owns the Keras import (`predict.py:16`), so
the boundary is unchanged. Note it returns *scaled* predictions; `permutation_importance` compares
against scaled `y` from `transform`, and R² is invariant under the shared affine transform, so no
inverse is needed — and skipping it avoids a needless round-trip per permutation.

#### 3. Tests

**File**: `apps/backend/tests/test_importance.py` (new)

**Intent**: Exercise the full loop without TensorFlow, then confirm the real model path once.

**Contract**: Fast tests using a stub scorer with known behaviour — a scorer reading exactly one
feature must rank that column top and the others near zero; a constant scorer must produce
near-zero importance everywhere. Assert determinism under a fixed seed, that `n_repeats` controls
`len(scores)`, that negative values survive unclamped, that a categorical column appears once rather
than once per one-hot level, and that a frame without the target column raises `SchemaError`. Add
one `@pytest.mark.slow` test scoring `OUTPUTS/example` with a real loaded model end to end.

**File**: `apps/backend/src/mlpp/__init__.py`

**Intent**: The module docstring enumerates which modules are TensorFlow-free; `importance` belongs
in that list and its absence would mislead.

**Contract**: Add `importance` to the TensorFlow-free enumeration in the docstring. No new exports.

### Success Criteria:

#### Automated Verification:

- Importance tests pass: `uv run pytest tests/test_importance.py -v`
- The fast subset of them runs without TensorFlow: `uv run pytest tests/test_importance.py -m 'not slow'`
- Fast suite stays TensorFlow-free: verify `tensorflow` and `keras` absent from `sys.modules` after `pytest -m 'not slow'`
- Fast suite runtime has not materially regressed (baseline 1.56s)
- Full suite passes: `uv run pytest`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`

#### Manual Verification:

- Importance on `OUTPUTS/example` ranks plausibly against the known feature set
- Re-running with the same seed reproduces identical numbers
- The `progress` callback fires the expected number of times

**Implementation Note**: Pause for manual confirmation before proceeding.

---

## Phase 4: Dashboard shell & introspection

### Overview

The interface appears. Session discovery, selection, error surfacing, and the four read-only
introspection panels. By this point importance and Parquet already work as libraries, so this phase
is assembly.

### Changes Required:

#### 1. Package scaffold and entrypoint

**File**: `apps/backend/src/mlpp/dashboard/__init__.py` (new)

**Intent**: Establish the subpackage on the TensorFlow-owning side of the module split, with a
docstring stating that explicitly so the boundary is discoverable from inside.

**Contract**: Docstring only; no re-exports that would pull Keras in at import time.

**File**: `apps/backend/src/mlpp/dashboard/app.py` (new)

**Intent**: The single entrypoint. Owns page config, the sidebar session selector, and dispatch to
panels. Holds no domain logic.

**Contract**: Run as `uv run --group dashboard streamlit run src/mlpp/dashboard/app.py` from
`apps/backend/`. A sidebar selector lists candidate directories under `OUTPUTS/`; changing it
re-derives every panel (FR-003) via Streamlit's natural rerun, with no manual invalidation. Panels
are dispatched through tabs. Every panel call is wrapped so `MlppError` renders through
`st.error(str(exc))` (FR-002) — the only place that catch belongs.

#### 2. Cached loaders

**File**: `apps/backend/src/mlpp/dashboard/loaders.py` (new)

**Intent**: Centralise every artifact read and every cache decision, so panels receive
already-loaded objects and no panel ever touches a filename. Streamlit reruns the whole script on
each interaction, so uncached loading would re-read the model on every click.

**Contract**: `list_sessions(root)` returns candidate directories sorted newest-first — a directory
is a candidate if it exists, not if it is valid; validation is `load_session`'s job and its failure
is what FR-002 renders. `load_session_cached(session_dir)` wraps `session.load_session` under
`@st.cache_data`. `load_model_cached(session_dir)` wraps `predict.load_model` under
`@st.cache_resource` — a Keras model is unhashable, which is what distinguishes the two decorators
here. `compute_importance_cached(session_dir, dataset_path, seed, n_repeats)` wraps
`importance.permutation_importance` under `@st.cache_data`, keyed on those four scalars (FR-010) —
never on a mutable object. Artifact filenames are resolved only via `manifest.filenames_for(ROLE_*)`
(FR-020).

#### 3. Introspection panels

**File**: `apps/backend/src/mlpp/dashboard/panels/introspect.py` (new)

**Intent**: Render architecture, feature contract, training history, metrics and inventory from an
already-loaded session. FR-004 through FR-007.

**Contract**: One `render(session, model)` per panel group, each taking loaded objects rather than
paths. Architecture reads `model.summary()` output; the feature contract reads
`manifest.features`; history and metrics read the `stage_history` / `training_log` / `metrics`
roles through the manifest; inventory lists every `ArtifactEntry` with on-disk presence confirmed.
Nothing here writes.

#### 4. Tests

**File**: `apps/backend/tests/test_dashboard.py` (new)

**Intent**: Cover the logic that is testable without a browser — session listing and the error
path — and leave rendering to manual verification.

**Contract**: Fast tests for `list_sessions` ordering and for its behaviour on an empty or absent
root. A test that a directory missing `manifest.json` surfaces `ArtifactError` and one that a
version mismatch surfaces `SchemaVersionError`, both with messages naming the file — the three
incomplete directories under `OUTPUTS/` are the real-world case this protects. Panel render tests
are `slow` where they need a model.

### Success Criteria:

#### Automated Verification:

- Dashboard tests pass: `uv run pytest tests/test_dashboard.py -v`
- Full suite passes: `uv run pytest`
- Fast suite stays TensorFlow-free: `uv run pytest -m 'not slow'`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- No filename literal appears in `dashboard/`: grep for `.keras`, `.gz`, `.csv` string literals returns nothing outside comments

#### Manual Verification:

- The dashboard starts and is reachable only from the local machine
- Selecting `OUTPUTS/example` shows architecture, contract, history and metrics
- Selecting one of the three incomplete sessions shows a message naming what is missing, not a traceback
- Switching sessions updates every panel without a restart
- Model load shows visible progress rather than an apparently frozen page

**Implementation Note**: Pause for manual confirmation before proceeding.

---

## Phase 5: Inference panels

### Overview

The two interactive surfaces: a single-row form with an out-of-range warning, and batch inference
over CSV or Parquet with download and unseen-level reporting. Plus the importance panel, which
consumes Phase 3's library.

### Changes Required:

#### 1. Importance panel

**File**: `apps/backend/src/mlpp/dashboard/panels/importance.py` (new)

**Intent**: Let the author pick a dataset, run importance against the selected session, and read the
result as a ranked chart. FR-008 through FR-010.

**Contract**: `render(session, model, dataset_path)`. Calls `compute_importance_cached`, passing a
`progress` callback bound to `st.progress` (FR-010). Renders a horizontal bar chart sorted by
`mean_drop` with `std_drop` as error bars, displaying negative values as-is. Seed and repeat count
are exposed as controls, defaulting to `seed=0`, `n_repeats=5`.

#### 2. Single-row inference

**File**: `apps/backend/src/mlpp/dashboard/panels/single.py` (new)

**Intent**: Score one hand-entered row without writing a file, and warn when an entered value falls
outside the range the scaler observed. FR-012, FR-013, FR-022.

**Contract**: `render(session, model)`. Form fields are generated from
`session.manifest.features` — numeric columns get number inputs, categorical columns get selectboxes
populated from `preprocessor.fitted_categories` (FR-013; no hardcoded column list). On submit,
assembles a one-row DataFrame and calls `predict.score_frame`, displaying the result in engineering
units. For FR-022, reads `mean_` and `scale_` off the fitted `StandardScaler` via
`preprocessor.fitted_state` and flags any numeric entry beyond a fixed number of standard
deviations from the fitted mean, naming the column and the observed range. Writes nothing to disk.

#### 3. Batch inference

**File**: `apps/backend/src/mlpp/dashboard/panels/batch.py` (new)

**Intent**: Score a whole CSV or Parquet file, report unseen categorical levels, and offer the
result as a download. FR-014 through FR-016.

**Contract**: `render(session, model)`. Reads the chosen file through `data.read_table_auto` (Phase
2), scores via `predict.score_frame`, and renders a preview. When `PredictionResult.has_unseen`, it
surfaces `describe_unseen()` as a warning naming column and values (FR-016) — the CLI emits this
only as a log line. Download goes through `st.download_button` with a toggle for including input
columns alongside the prediction column (FR-015). Nothing is written to `OUTPUTS/`.

#### 4. Tests

**File**: `apps/backend/tests/test_dashboard.py`

**Intent**: Cover the non-rendering logic these panels add — the range check and the download frame
shape — in the fast suite.

**Contract**: Fast tests for the out-of-range detection given a fitted scaler's `mean_`/`scale_`,
including the boundary case exactly at the threshold. Fast test that the download frame carries the
prediction column alone when the toggle is off and inputs plus prediction when on. Scoring paths
that need a real model are `slow`.

### Success Criteria:

#### Automated Verification:

- Dashboard tests pass: `uv run pytest tests/test_dashboard.py -v`
- Full suite passes: `uv run pytest`
- Fast suite stays TensorFlow-free and has not materially regressed: `uv run pytest -m 'not slow'`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- Nothing under `OUTPUTS/` changed during a full manual session: `git status --porcelain OUTPUTS/` is empty

#### Manual Verification:

- Importance against a TEST file produces a ranked chart with visible error bars and progress
- Re-running the same session, dataset and seed returns instantly from cache
- The single-row form generates fields matching the session contract, and an extreme value triggers the range warning
- Batch inference over both a CSV and a Parquet file produces predictions and a working download
- A file with an unseen categorical level surfaces the warning naming column and values
- All four PRD surfaces are reachable from one `streamlit run` without returning to a terminal

**Implementation Note**: This is the final phase. After manual confirmation, the change is ready to
archive.

---

## Testing Strategy

### Unit Tests:

- `read_table_auto` routing, both suffixes, and `DatasetError` on corrupt or empty Parquet
- `permutation_importance` against stub scorers with known behaviour: single-feature, constant,
  determinism under seed, negative values unclamped, categorical aggregation, missing target
- `list_sessions` ordering and empty/absent root
- Session error surfacing: missing manifest, version mismatch, manifest listing absent files
- FR-022 range detection including the exact-threshold boundary
- Download frame shape with the input-columns toggle both ways

### Integration Tests:

- `mlpp-predict` end to end against both a CSV and a Parquet input, asserting CSV output is
  byte-identical to the pre-change result
- `permutation_importance` against `OUTPUTS/example` with a real loaded model (`slow`)

### Manual Testing Steps:

1. Start the dashboard; confirm it is reachable only from the local machine.
2. Select `OUTPUTS/example`; confirm architecture, contract, history and metrics all render.
3. Select `OUTPUTS/2026-07-30_12-40-12` (incomplete); confirm a message names what is missing.
4. Switch back; confirm every panel re-derives without a restart.
5. Run importance against a TEST file; confirm progress, ranked chart, and error bars.
6. Re-run with the same seed; confirm it returns from cache instantly and the numbers are identical.
7. Enter a row in the single-row form; confirm a prediction and an out-of-range warning on an
   extreme value.
8. Score a CSV and a Parquet file in batch; download both, with and without input columns.
9. Confirm `git status --porcelain OUTPUTS/` is empty afterwards.

## Performance Considerations

Importance is the only expensive operation: `len(schema.inputs) × n_repeats` transform-and-score
passes, plus one baseline. At the current 13 columns and 5 repeats that is 66 passes over a
test-sized frame — seconds. It grows linearly in both columns and repeats, which is why FR-010
mandates caching and progress rather than treating the cheap case as the guaranteed one. Model load
is a one-off few seconds, absorbed by `@st.cache_resource`.

## Migration Notes

None. No artifact is rewritten, moved, or reinterpreted; `SCHEMA_VERSION` stays at 1. The three
incomplete session directories are left exactly as they are and serve as FR-002's test cases.

Rollback is per-phase: Phases 2–5 are additive, so reverting any commit leaves the previous phase
working. Phase 1's instruction-file edits are documentation and carry no runtime effect.

## References

- PRD: `context/foundation/prd.md`
- Shape notes: `context/foundation/shape-notes.md`
- Stack assessment: `context/foundation/stack-assessment.md` — paste-ready instruction-file text
- Health check: `context/foundation/health-check.md` — the joblib/numpy finding behind Phase 1
- Artifact contract: `apps/backend/src/mlpp/session.py:36-53`
- One-hot expansion order: `apps/backend/src/mlpp/preprocess.py:199-200`
- Unseen-level reporting: `apps/backend/src/mlpp/predict.py:34-51`
- Reader call sites: `apps/backend/src/mlpp/predict_cli.py:106`, `pipeline.py:90`, `pipeline.py:193`

## Progress

> Convention: `- [ ]` pending, `- [x]` done. Append ` — <commit sha>` when a step lands. Do not rename step titles. See `references/progress-format.md`.

### Phase 1: Prerequisites & contract

#### Automated

- [x] 1.1 Full suite passes: `cd apps/backend && uv run pytest` — f91206d
- [x] 1.2 Fast suite still TensorFlow-free: `uv run pytest -m 'not slow'` — f91206d
- [x] 1.3 Lockfile resolves and is not stale: `uv lock --check` — f91206d
- [x] 1.4 Type checking passes: `uv run mypy src/mlpp` — f91206d
- [x] 1.5 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — f91206d
- [x] 1.6 `pyarrow` importable from a base install: `uv run --no-group dashboard python -c "import pyarrow"` — f91206d
- [x] 1.7 The joblib tripwire fires on an unignored deprecation — f91206d

#### Manual

- [x] 1.8 No sentence in `CLAUDE.md` or `.cursorrules` now forbids the dashboard — f91206d
- [x] 1.9 The `## Dashboard conventions` section describes the layout Phases 4–5 will build — f91206d
- [x] 1.10 The scoped joblib ignore names the condition and points at the follow-up change — f91206d

### Phase 2: Parquet reader

#### Automated

- [x] 2.1 New data tests pass: `uv run pytest tests/test_data.py -v` — 0b87e6d
- [x] 2.2 CLI tests pass: `uv run pytest tests/test_predict_cli.py -v` — 0b87e6d
- [x] 2.3 Full suite passes: `uv run pytest` — 0b87e6d
- [x] 2.4 Fast suite stays TensorFlow-free and ~1.5s: `uv run pytest -m 'not slow'` — 0b87e6d
- [x] 2.5 Type checking passes: `uv run mypy src/mlpp` — 0b87e6d
- [x] 2.6 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — 0b87e6d
- [x] 2.7 Training path untouched: `read_csv_auto` still has exactly two call sites in `pipeline.py` — 0b87e6d

#### Manual

- [x] 2.8 `mlpp-predict` scores a Parquet input correctly — 0b87e6d
- [x] 2.9 `mlpp-predict` against `TEST/sample_test_A.csv` produces unchanged output — 0b87e6d

### Phase 3: Feature importance

#### Automated

- [x] 3.1 Importance tests pass: `uv run pytest tests/test_importance.py -v` — ae29a3a
- [x] 3.2 Fast subset runs without TensorFlow: `uv run pytest tests/test_importance.py -m 'not slow'` — ae29a3a
- [x] 3.3 Fast suite leaves `tensorflow` and `keras` absent from `sys.modules` — ae29a3a
- [x] 3.4 Fast suite runtime has not materially regressed (baseline 1.56s) — ae29a3a
- [x] 3.5 Full suite passes: `uv run pytest` — ae29a3a
- [x] 3.6 Type checking passes: `uv run mypy src/mlpp` — ae29a3a
- [x] 3.7 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — ae29a3a

#### Manual

- [x] 3.8 Importance on `OUTPUTS/example` ranks plausibly against the known feature set — ae29a3a
- [x] 3.9 Re-running with the same seed reproduces identical numbers — ae29a3a
- [x] 3.10 The `progress` callback fires the expected number of times — ae29a3a

### Phase 4: Dashboard shell & introspection

#### Automated

- [x] 4.1 Dashboard tests pass: `uv run pytest tests/test_dashboard.py -v` — 5fcc11d
- [x] 4.2 Full suite passes: `uv run pytest` — 5fcc11d
- [x] 4.3 Fast suite stays TensorFlow-free: `uv run pytest -m 'not slow'` — 5fcc11d
- [x] 4.4 Type checking passes: `uv run mypy src/mlpp` — 5fcc11d
- [x] 4.5 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — 5fcc11d
- [x] 4.6 No filename literal appears in `dashboard/` — 5fcc11d

#### Manual

- [x] 4.7 The dashboard starts and is reachable only from the local machine — 5fcc11d
- [x] 4.8 Selecting `OUTPUTS/example` shows architecture, contract, history and metrics — 5fcc11d
- [x] 4.9 An incomplete session shows a message naming what is missing, not a traceback — 5fcc11d
- [x] 4.10 Switching sessions updates every panel without a restart — 5fcc11d
- [x] 4.11 Model load shows visible progress rather than an apparently frozen page — 5fcc11d

### Phase 5: Inference panels

#### Automated

- [x] 5.1 Dashboard tests pass: `uv run pytest tests/test_dashboard.py -v` — a609721
- [x] 5.2 Full suite passes: `uv run pytest` — a609721
- [x] 5.3 Fast suite stays TensorFlow-free and has not materially regressed — a609721
- [x] 5.4 Type checking passes: `uv run mypy src/mlpp` — a609721
- [x] 5.5 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — a609721
- [x] 5.6 `git status --porcelain OUTPUTS/` is empty after a full manual session — a609721

#### Manual

- [x] 5.7 Importance produces a ranked chart with error bars and visible progress — a609721
- [x] 5.8 Re-running the same session, dataset and seed returns instantly from cache — a609721
- [x] 5.9 The single-row form matches the session contract and warns on an extreme value — a609721
- [x] 5.10 Batch inference works over both CSV and Parquet, with a working download — a609721
- [x] 5.11 An unseen categorical level surfaces a warning naming column and values — a609721
- [x] 5.12 All four PRD surfaces are reachable from one `streamlit run` — a609721

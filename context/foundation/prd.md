---
project: "mlpp Dashboard"
version: 1
status: draft
created: 2026-07-31
context_type: brownfield
product_type: web-app
target_scale:
  users: small
  qps: negligible
  data_volume: small
timeline_budget:
  delivery_weeks: 3
  hard_deadline: null
  after_hours_only: true
---

# PRD — mlpp Dashboard

## Current System Overview

**Purpose.** A time-series / tabular regression pipeline that trains a Conv1D → Bi-GRU → Dense
model sequentially over `TRAIN/*.csv`, evaluates it against `TEST/*.csv`, and writes a timestamped,
manifest-governed artifact directory under `OUTPUTS/`.

**Architecture.** A single Python package (`mlpp`, src layout) under `apps/backend/`, driven by two
console entrypoints. Deliberately split into a TensorFlow-free half (`config`, `data`, `preprocess`,
`metrics`, `session`, `errors`) and a TensorFlow-owning half (`model`, `training`, `plots`,
`pipeline`, `predict`). No frontend, no service, no infrastructure. The product type today is a
data-pipeline with command-line entrypoints; this change adds a local web-app surface alongside it
without changing what the pipeline itself is.

**Tech stack.** Python 3.12–3.13, uv (lockfile `apps/backend/uv.lock`), TensorFlow 2.20–2.21,
Keras 3, numpy 2.x, pandas 2.2–3.x, scikit-learn 1.5+, joblib, matplotlib, plotly. Tooling: pytest
(with a `slow` marker for TensorFlow tests), ruff, mypy strict.

**Current user base.** One person: the model author, working locally from a terminal.

**Core functionality today.**
- `mlpp-train` — sequential fit over training CSVs, evaluation, artifact session write.
- `mlpp-predict` — score one CSV against a saved session, write a predictions CSV.
- `notebooks/01_exploration_and_baseline.ipynb` — a static demonstration report reading the
  committed `OUTPUTS/example/` session. Contains no pipeline logic by design.
- Post-hoc inspection is file-based: open a `prediction_analysis_*.html`, a `training_curves_*.png`,
  or read `test_metrics.csv` / `training_log.csv` by hand.

**Artifact contract.** Each run produces `OUTPUTS/<YYYY-MM-DD_HH-MM-SS>/`, git-ignored except the
committed `OUTPUTS/example/`. `manifest.json` (`schema_version: 1`) is the single source of truth
for the column contract and the file inventory; `session.py` is its sole owner. Artifact roles are
`best_model`, `stage_model`, `stage_history`, `training_log`, `scaler`, `output_scaler`, `encoders`,
`training_curves`, `prediction_analysis` and `metrics`. Eight session directories exist on disk
today; three of them are incomplete, carrying models but no metrics or plots.

## Problem Statement & Motivation

Inspecting a trained model is a file-archaeology exercise. To understand a session the author opens
a directory of CSVs, PNGs and standalone HTML reports and reconstructs the picture manually; to
score anything they compose an `mlpp-predict` invocation with three path arguments. Four things are
either impossible or disproportionately expensive today:

1. **Model introspection** — no view of architecture, the feature contract, or training history
   beyond raw CSVs.
2. **Global feature importance** — not computable at all; nothing in the package measures which
   inputs drive predictions.
3. **Real-time single inference** — there is no way to enter one row of feature values and see a
   prediction. The existing path requires writing a CSV to disk first.
4. **Batch inference on a chosen file** — possible for CSV from the command line, but there is no
   Parquet path and no interactive file selection.

**Why now.** The pipeline itself is stable and consolidated — the preceding change closed out the
notebook work — so the remaining friction is entirely in *reading* what the pipeline produced rather
than in producing it.

**Current workaround and its cost.** The author opens `training_curves_*.png`, `test_metrics.csv`
and `history_train_*.csv` separately and reconstructs the picture by hand; for scoring, they write a
CSV to disk and compose a three-argument command. For feature importance there is no workaround at
all.

## User & Persona

**Primary persona: the model author.** The single technical user who trains these models, runs the
existing command-line entrypoints, and owns the repository. They are comfortable in a terminal — the
dashboard is not a literacy workaround but a speed and comprehension tool. They work locally on one
machine, against session directories on the same disk.

**Whose experience changes.** This is the only existing user, and all four friction points above are
theirs. No new users are enabled by this change; there is no secondary persona in scope.

## Success Criteria

### Primary
- From a single command, the author can pick any session directory under `OUTPUTS/` and, without
  returning to a terminal, see its architecture and feature contract, read a global
  feature-importance ranking, score one hand-entered row, and score a whole CSV or Parquet file.
- Every one of those reads resolves artifacts through `manifest.json` roles — no filename is spelled
  outside `session.py`.

### Secondary
- Feature importance is available outside the dashboard too — importable, and cheap to expose from a
  command-line entrypoint later — rather than being trapped in interface code.
- A malformed or incomplete session directory (three exist under `OUTPUTS/` today) produces a clear
  in-app error naming the problem, instead of a stack trace.

### Guardrails
- `uv run pytest -m 'not slow'` stays TensorFlow-free and stays at roughly its current ~1s runtime.
  No dashboard code may pull the training stack into the fast path.
- `config`, `data`, `preprocess`, `metrics`, `session` and `errors` continue to import no
  TensorFlow.
- `session.py` remains the sole owner of every filename in a session directory.
- `mlpp-train` and `mlpp-predict` keep their current behavior and flags; this change is additive.
- `SCHEMA_VERSION` stays at 1 and `OUTPUTS/example/` keeps loading unchanged.
- `uv run mypy src/mlpp` and `ruff check` / `ruff format --check` stay clean, for the new code as
  for the rest.
- Any operation taking longer than two seconds — model load, importance computation — shows
  continuous visible progress. The interface never appears unresponsive.
- Every failure reaching the author names the artifact and the reason — a missing file, a schema
  version, an unparseable input — and never surfaces as a bare traceback. Errors originate from the
  `MlppError` hierarchy.

## User Stories

Delta-framed against today's command-line-only workflow.

### US-01: Author inspects a trained session without opening a file

- **Given** at least one session directory exists under `OUTPUTS/`
- **When** the author launches the dashboard and selects a session
- **Then** they see its architecture, its feature contract, its training history and its test metrics
  in one place

*Before:* the author opened `training_curves_*.png`, `test_metrics.csv` and `history_train_*.csv`
separately and reconstructed the picture by hand.

#### Acceptance Criteria
- Every displayed artifact was resolved through a `manifest.json` role, not a hardcoded filename.
- Selecting one of the three incomplete sessions produces a message naming what is missing, not a
  traceback.
- Switching sessions updates every panel without a restart.

### US-02: Author learns which inputs the model actually depends on

- **Given** a selected session and a dataset that carries the target column
- **When** the author starts an importance run
- **Then** they see a ranked chart of input columns by mean R² degradation, with the spread across
  repeats shown alongside each column

*Before:* impossible — no such capability existed.

#### Acceptance Criteria
- Ranking is per input column; one-hot-expanded columns are aggregated back to their origin.
- Progress is visible while the run is in flight.
- Re-selecting the same session, dataset and seed reuses the completed result.
- A negative degradation is displayed as-is, not clamped to zero.

### US-03: Author scores one hand-entered row

- **Given** a selected session
- **When** the author fills the input form and submits
- **Then** they see a prediction in engineering units, and a warning on any field whose value falls
  outside the range the scaler observed during fitting

*Before:* required writing a one-row CSV to disk and composing an `mlpp-predict` invocation.

#### Acceptance Criteria
- Form fields are generated from the session's feature contract.
- No file is written anywhere, including inside the session directory.

### US-04: Author scores a whole file

- **Given** a selected session
- **When** the author chooses a CSV or Parquet file
- **Then** they see predictions for every row and can download them, optionally with the input
  columns alongside

*Before:* CSV only, from the command line, with no Parquet path.

#### Acceptance Criteria
- Unrecognised categorical levels are reported in the interface, naming column and values.
- The download leaves through the browser; `OUTPUTS/` is not written to.

## Scope of Change

Format: `FR-NNN: [Actor] can [capability]. Priority. Change: new | modified | preserved`.
The actor is the model author throughout — there is only one persona.

### Session selection

- FR-001: Author can list every session directory under `OUTPUTS/` and select one to inspect.
  Priority: must-have. Change: new
- FR-002: Author sees an in-app error naming the specific problem when a selected directory is not a
  valid session — missing manifest, schema-version mismatch, or manifest listing files absent from
  disk. Priority: must-have. Change: new
- FR-003: Author can switch the selected session without restarting the app, and every panel
  re-derives from the newly selected session. Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "Session selection is chrome; just take a path
  > argument." Resolution: kept. Eight session directories exist and three are broken; discovery
  > plus a naming error message is the cheapest path to the Secondary success criterion.

### Model introspection

- FR-004: Author can view the trained model's architecture — layers, output shapes, parameter
  counts. Priority: must-have. Change: new
- FR-005: Author can view the session's feature contract as recorded in `manifest.json`: numeric
  columns, categorical columns, output column, and the post-one-hot `feature_names` order.
  Priority: must-have. Change: new
- FR-006: Author can view per-stage training history and the recorded test metrics without opening
  a CSV by hand. Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "This duplicates the notebook report." Resolution: kept.
  > The notebook is a static demonstration bound to `OUTPUTS/example/`; these panels are live against
  > any session.

- FR-007: Author can view the session's inventory — every artifact role and filename the manifest
  records, with presence confirmed on disk. Priority: nice-to-have. Change: new

  > Socrates: Counter-argument considered — "Inventory display is developer trivia." Resolution:
  > kept at nice-to-have — useful for diagnosing the broken sessions, not required for the Primary
  > criterion.

### Global feature importance

- FR-008: Author can compute global permutation feature importance for the selected session against
  a dataset they choose. Priority: must-have. Change: new
- FR-009: Author sees importance reported per input column, with one-hot-expanded columns aggregated
  back to the originating categorical column. Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "Aggregation is a no-op: `categorical_columns` is empty
  > in every session that exists." Resolution: kept as must-have. The pipeline supports categoricals,
  > and importance that reported a one-hot column instead of its originating column would be silently
  > wrong the first time one is configured. Correctness over YAGNI; the cost is small.

- FR-010: Author sees progress while an importance run is in flight, and a completed run is reused
  rather than recomputed when the same session, dataset and seed are re-selected.
  Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "With 13 features and a small test file, permutation is
  > seconds; caching and progress reporting are premature." Resolution: kept as written, both parts.
  > Feature count and row count are both config-driven, so the cheap case is not the guaranteed case.

- FR-011: Importance computation is importable from the package independently of the dashboard, so
  it is not trapped in interface code. Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "Importance belongs in the dashboard; a separate module
  > is over-engineering." Resolution: kept. Trapping it in interface code contradicts the Secondary
  > criterion and makes it untestable in the fast suite.

### Single-row inference

- FR-012: Author can enter a value for each input column in a form and receive a prediction in
  engineering units without writing a file to disk. Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "One hand-typed row yields a confident number with no
  > signal about whether the input is in-distribution." Resolution: kept, and hardened — FR-022
  > added. The fitted scaler is already loaded, so an out-of-range check is nearly free and removes
  > the footgun.

- FR-013: The form's fields are generated from the selected session's feature contract, never from a
  hardcoded column list. Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "Hardcoding 13 fields is faster." Resolution: kept.
  > Hardcoding a column list is exactly the drift that the manifest-as-single-source-of-truth
  > invariant exists to prevent.

- FR-022: Author is warned when a value entered in the single-row form falls outside the range the
  scaler observed during fitting. Priority: must-have. Change: new

### Batch inference

- FR-014: Author can select a CSV or Parquet file and receive predictions for every row.
  Priority: must-have. Change: new
- FR-015: Author can download the batch predictions, with the option to include the input columns
  alongside the prediction column. Priority: must-have. Change: new
- FR-016: Author is told when input rows carried categorical levels the encoder never saw, including
  which column and which values. Priority: must-have. Change: new

  > Socrates: Counter-argument considered — "`mlpp-predict` already does batch." Resolution: kept.
  > It requires composing three paths in a terminal and cannot read Parquet; FR-016 surfaces the
  > unseen-level report, which the existing entrypoint emits only as a log line.

- FR-017: `mlpp` can read a Parquet file wherever it currently reads a CSV for scoring input.
  Priority: must-have. Change: modified

  > Socrates: Counter-argument considered — "`data.py` is a preserved TensorFlow-free module on the
  > training path; a dashboard should not modify it." Resolution: kept, but constrained to additive.
  > A new suffix-dispatching reader is added alongside `read_csv_auto`, which is not itself changed.
  > The training path keeps calling the existing function and therefore cannot regress, while Parquet
  > becomes available to the whole package rather than to the interface alone.

### Explicitly preserved

- FR-018: `mlpp-train` and `mlpp-predict` continue to work with their current behavior and flags.
  Priority: must-have. Change: preserved
- FR-019: `config`, `data`, `preprocess`, `metrics`, `session` and `errors` continue to import no
  TensorFlow, and `uv run pytest -m 'not slow'` stays TensorFlow-free. Priority: must-have.
  Change: preserved
- FR-020: `session.py` remains the sole owner of every filename in a session directory; the
  dashboard resolves artifacts only through manifest roles. Priority: must-have. Change: preserved
- FR-021: `SCHEMA_VERSION` stays at 1 and `OUTPUTS/example/` continues to load unchanged.
  Priority: must-have. Change: preserved

  > Socrates: FR-018 through FR-021 are preserved-behavior requirements. No counter-argument was
  > raised: these are the change's guardrails and were selected explicitly by the author.

### Nothing is removed

No existing capability is withdrawn by this change.

## Constraints & Compatibility

**Backward compatibility.**
- `manifest.json` layout is unchanged; `SCHEMA_VERSION` stays at 1. No new artifact role is
  introduced, which is why the importance dataset is author-selected rather than persisted.
- `mlpp-train` and `mlpp-predict` keep their current flags and behavior. Neither is a dependency of
  the dashboard's correctness.
- `read_csv_auto` is not modified. Parquet support arrives as an additive sibling reader, so the
  training path's behavior is untouched by construction.

**Data migration.** None. No existing artifact is rewritten, moved, or reinterpreted. The three
incomplete session directories currently under `OUTPUTS/` are not repaired — they become test cases
for FR-002's error reporting.

**Existing integrations that must keep working.**
- The committed `OUTPUTS/example/` session, which the notebook report reads and the drift gate
  checks.
- `notebooks/01_exploration_and_baseline.ipynb` and its generator, `scripts/build_notebook.py`.
- The existing development and notebook dependency groups. The new work is grouped separately so
  that a sync without it still yields a working train/predict install.

**Preserved behavior, explicitly.**
- The TensorFlow-free module set (`config`, `data`, `preprocess`, `metrics`, `session`, `errors`)
  imports no TensorFlow. The new dashboard code sits deliberately on the TensorFlow-owning side of
  that line, alongside `model`, `training`, `plots`, `pipeline` and `predict`.
- `session.py` remains the sole owner of session filenames.
- Sessions are never written to. The dashboard has read-only access to `OUTPUTS/`.

## Business Logic Changes

**This change adds one new domain rule.** It is not, as originally scoped, purely an interface over
existing capability — three of the four surfaces are wrappers, but feature importance is a decision
the system makes, and nothing in `mlpp` makes it today.

**The rule, in one sentence:** the importance of an input column is how much the model's accuracy
degrades when that column's values are randomly shuffled and everything else is held fixed.

The rule consumes a trained session and a dataset the author selects that carries the target column.
It scores the dataset once to establish a baseline, then, for each input column in turn, shuffles
that column's values across rows, re-scores, and records the loss of accuracy. Accuracy is measured
as **R²**, so importance reads as "shuffling this column costs the model N points of R²" and stays
comparable across datasets and sessions with different target scales. Shuffling is **repeated
several times per column with a fixed seed**, and both the mean degradation and its spread are
reported — a ranking difference smaller than the spread is noise, and the author must be able to see
that rather than infer a false ordering.

Two properties of this pipeline shape the rule. First, one-hot expansion means the model's input
axis does not correspond to input columns; a categorical column occupies several positions, and all
of them must be shuffled together for the result to describe the column rather than a level.
Second, a column that the model genuinely ignores can score slightly negative — shuffling it happens
to help — and that is a meaningful signal, not an error to clamp away.

The author encounters the rule as a ranked chart, after selecting a session and a dataset. No other
domain logic changes.

# TODO: the number of permutation repeats is unspecified — see Open Questions

## Access Control Changes

The current system has no access control: two console entrypoints run by whoever holds a shell on
the machine. This change does not alter that model, but it does introduce a **new exposure surface**
the existing entrypoints never had — a long-running local process reachable over a network port.

Decision: **local machine only, no authentication.** The dashboard is reachable only from the
machine it runs on. No accounts, no roles, no secrets. This is an operating constraint rather than
something enforced in code; a startup guard was considered and deferred.

No roles exist and none are introduced.

## Non-Goals

- **No training from the dashboard.** The interface reads sessions; it never launches a run.
  Training has long-running, checkpointing, sequential-fit semantics that belong to `mlpp-train` and
  would dominate the delivery budget if surfaced.
- **No gradient- or game-theoretic attribution methods.** Permutation importance is the only method
  in scope. This also rules out per-row local explanations — the change delivers *global* importance
  only.
- **No hosting, deployment, or multi-user access.** No container image, no auth, no shared instance,
  no concurrency handling. Local machine only, run by hand.
- **No session mutation or cleanup tooling.** The dashboard will not delete, repair, rename, or
  regenerate sessions — including the three incomplete directories currently under `OUTPUTS/`. Those
  exist to be *reported* on, not fixed. Read-only, without exception.
- **No new artifact role and no `SCHEMA_VERSION` bump.** A non-goal because it was a live option:
  persisting an importance sample at train time was considered during shaping and rejected precisely
  because it would breach this line.

## Open Questions

1. **How many permutation repeats is the default?** Shaping settled on "several, seeded, mean and
   spread reported" but not the number. It affects both run duration and how trustworthy a narrow
   ranking gap is. Owner: author. Resolve during planning. Block: no.
2. **Should `mlpp-predict` gain Parquet support in this change or later?** FR-017 places the reader
   in `data.py`, which makes wiring it into the existing predict entrypoint nearly free — but that
   entrypoint is marked preserved under FR-018, so extending it is a deliberate scope decision
   rather than a free consequence. Owner: author. Resolve during planning. Block: no.

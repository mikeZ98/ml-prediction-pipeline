---
project: "mlpp Dashboard"
context_type: brownfield
created: 2026-07-31
updated: 2026-07-31
checkpoint:
  current_phase: 8
  phases_completed: [1, 2, 3, 4, 5, 6, 7]
  gray_areas_resolved:
    - topic: "primary persona"
      decision: "the model author — single technical user, local, no auth"
    - topic: "change category"
      decision: "new module: a UI surface over existing capability"
    - topic: "preserved behavior"
      decision: "all four: TF-free split, session.py filename ownership, both CLIs, schema_version 1 + OUTPUTS/example"
    - topic: "access / exposure"
      decision: "localhost-only, no auth; startup guard considered and deferred"
    - topic: "MVP scope"
      decision: "all four surfaces in v1; importance constrained to permutation-based (no SHAP)"
    - topic: "timeline"
      decision: "3 delivery weeks, after hours"
    - topic: "importance dataset source"
      decision: "author picks a file in the dashboard; no new artifact role, no train-CLI change"
    - topic: "dashboard code placement"
      decision: "apps/backend/src/mlpp/dashboard/ — a subpackage, explicitly outside the TF-free rule"
    - topic: "parquet dependency"
      decision: "declare pyarrow as a direct dependency rather than relying on streamlit's transitive pull"
    - topic: "parquet reader boundary"
      decision: "additive new reader in data.py; read_csv_auto left untouched"
  frs_drafted: 22
  quality_check_status: accepted
---

# Shape notes — s-06 (Interactive dashboard)

Discovery complete (phases 1–7). Ready for `/10x-prd`, which maps the sections below onto the
11-section brownfield PRD template. The `## Audit findings` and `## Forward: technical roadmap`
blocks sit outside the PRD schema and do not flow into `prd.md`.

## Audit findings (pre-phase, factual — from code inspection 2026-07-31)

Recorded here so later phases can reason against reality rather than assumption.

### Artifact directory

- The artifact root is `OUTPUTS/`, **not** `artifacts/`. No `artifacts/` directory exists.
- Layout: `OUTPUTS/<YYYY-MM-DD_HH-MM-SS>/` per training run; git-ignored except `OUTPUTS/example/`.
- 8 session directories on disk; 3 are incomplete (`2026-07-30_12-40-12`, `13-07-15`, `13-08-16`
  carry models but no metrics/plots — abandoned runs).
- `manifest.json` (`schema_version: 1`) is the single source of truth for the column contract and
  the file inventory. `session.py` is its sole owner.
- Artifact roles: `best_model`, `stage_model`, `stage_history`, `training_log`, `scaler`,
  `output_scaler`, `encoders`, `training_curves`, `prediction_analysis`, `metrics`.

### Dependency state (uv)

- `apps/backend/pyproject.toml` already carries an **uncommitted** `dashboard` dependency group
  pinning `streamlit>=1.40,<2`; `uv.lock` has resolved it (streamlit 1.60.0, +295 lock lines).
- `pyarrow` 24.0.0 enters the graph **transitively via streamlit** — Parquet reading is available
  but is not a declared direct dependency.
- No feature-importance library is present (no SHAP, captum, or `sklearn.inspection` usage).

### Capability gaps vs. the four requested surfaces

1. **Model introspection** — `predict.load_model()` exists; no architecture/layer summary helper.
2. **Global feature importance** — nothing exists. `X` is shaped `(rows, n_features, 1)` and the
   model is not an sklearn estimator, so `permutation_importance` needs a custom wrapper.
3. **Single-row inference** — `predict.score_frame()` covers it; TF import cost needs caching.
4. **Batch CSV/Parquet** — `data.read_csv_auto()` is CSV-only; `discover_csvs()` globs `*.csv`.
   No Parquet read path exists.

### Standing constraints observed

- `CLAUDE.md` declares the project "Python only — there is no frontend"; `apps/frontend/` is to be
  added only when something real goes in it.
- The TF-free / TF module split is load-bearing: `config`, `data`, `preprocess`, `metrics`,
  `session`, `errors` must not import TensorFlow (keeps the fast suite fast and lets
  `load_session` validate without the training stack).

---

## Current System

**Purpose.** A time-series / tabular regression pipeline that trains a Conv1D → Bi-GRU → Dense
model sequentially over `TRAIN/*.csv`, evaluates it against `TEST/*.csv`, and writes a timestamped,
manifest-governed artifact directory under `OUTPUTS/`.

**Architecture.** A single Python package (`mlpp`, src layout) under `apps/backend/`, driven by two
console entrypoints. Deliberately split into a TensorFlow-free half (`config`, `data`, `preprocess`,
`metrics`, `session`, `errors`) and a TensorFlow-owning half (`model`, `training`, `plots`,
`pipeline`, `predict`). No frontend, no service, no infrastructure.

**Tech stack.** Python 3.12–3.13, uv (lockfile `apps/backend/uv.lock`), TensorFlow 2.20–2.21,
Keras 3, numpy 2.x, pandas 2.2–3.x, scikit-learn 1.5+, joblib, matplotlib, plotly. Tooling: pytest
(with a `slow` marker for TF tests), ruff, mypy strict.

**Users today.** One person: the model author, working locally from a terminal.

**Core functionality today.**
- `mlpp-train` — sequential fit over training CSVs, evaluation, artifact session write.
- `mlpp-predict` — score one CSV against a saved session, write a predictions CSV.
- `notebooks/01_exploration_and_baseline.ipynb` — a static demonstration report reading the
  committed `OUTPUTS/example/` session. Contains no pipeline logic by design.
- Post-hoc inspection is file-based: open a `prediction_analysis_*.html`, a `training_curves_*.png`,
  or read `test_metrics.csv` / `training_log.csv` by hand.

## Problem Statement & Motivation

Inspecting a trained model is a file-archaeology exercise. To understand a session the author opens
a directory of CSVs, PNGs, and standalone HTML reports and reconstructs the picture manually; to
score anything they compose a `mlpp-predict` invocation with three path arguments. Four things are
either impossible or disproportionately expensive today:

1. **Model introspection** — no view of architecture, the feature contract, or training history
   beyond raw CSVs.
2. **Global feature importance** — not computable at all; nothing in the package measures which
   inputs drive predictions.
3. **Real-time single inference** — there is no way to type one row of feature values and see a
   prediction. The CLI requires writing a CSV to disk first.
4. **Batch inference on a chosen file** — possible for CSV via the CLI, but there is no Parquet
   path and no interactive file selection.

The trigger is that the pipeline itself is now stable and consolidated (s-05 closed the notebook
work), so the remaining friction is entirely in *reading* what the pipeline produced rather than in
producing it.

## User & Persona

**Primary persona: the model author.** The single technical user who trains these models, runs the
CLIs, and owns the repository. They are comfortable in a terminal — the dashboard is not a literacy
workaround but a speed and comprehension tool. They work locally on one machine, against session
directories on the same disk. There is no second persona in scope.

> Socrates (raised Phase 1, **resolved Phase 5**): the change was scoped as "a UI surface over
> existing capability", but global feature importance does not exist in `mlpp` in any form.
> Resolution: the scope is wider than a UI wrapper. Three of the four surfaces are wrappers over
> `load_session` / `score_frame`; feature importance is a new domain rule and is recorded as such in
> `## Business Logic Changes`.

## Access Control Changes

The current system has no access control: two console entrypoints run by whoever holds a shell on
the machine. This change does not alter that model, but it does introduce a **new exposure surface**
the CLIs never had — a Streamlit process binding a TCP port.

Decision: **localhost only, no authentication.** The dashboard binds the loopback interface and is
reachable only from the machine it runs on. No accounts, no roles, no secrets. This is documented as
an operating constraint rather than enforced in code; a startup guard was considered and deferred.

No roles exist and none are introduced.

## Success Criteria

### Primary
- From a single `streamlit run` command, the author can pick any session directory under `OUTPUTS/`
  and, without touching a terminal again, see its architecture and feature contract, read a global
  feature-importance ranking, score one hand-entered row, and score a whole CSV or Parquet file.
- Every one of those reads resolves artifacts through `manifest.json` roles — no filename is spelled
  outside `session.py`.

### Secondary
- Feature importance is available outside the dashboard too (importable, and cheap to expose from a
  CLI later) rather than being trapped in UI code.
- A malformed or incomplete session directory (three exist under `OUTPUTS/` today) produces a clear
  in-app error naming the problem, instead of a stack trace.

### Guardrails
- `uv run pytest -m 'not slow'` stays TensorFlow-free and stays at roughly its current ~1s runtime.
  No dashboard code may pull TF into the fast path.
- `config`, `data`, `preprocess`, `metrics`, `session`, `errors` continue to import no TensorFlow.
- `session.py` remains the sole owner of every filename in a session directory.
- `mlpp-train` and `mlpp-predict` keep their current behavior and flags; this change is additive.
- `SCHEMA_VERSION` stays at 1 and `OUTPUTS/example/` keeps loading unchanged.
- `uv run mypy src/mlpp` and `ruff check` stay clean.

## Timeline budget

`delivery_weeks: 3`, after-hours. Within the three-week gate, so no sustained-effort acknowledgment
is required — but the budget is binding on method choice: feature importance must be permutation-
based, computed from artifacts already on disk. SHAP, captum, or any gradient-attribution approach
is out of scope for this delivery.

## Scope of Change

Format: `FR-NNN: [Actor] can [capability]. Priority. Change: new | modified | preserved`.
Actor is the model author throughout — there is only one persona.

### Session selection

- FR-001: Author can list every session directory under `OUTPUTS/` and select one to inspect.
  Priority: must-have. Change: new
- FR-002: Author sees an in-app error naming the specific problem when a selected directory is not a
  valid session — missing manifest, schema-version mismatch, or manifest listing files absent from
  disk. Priority: must-have. Change: new
- FR-003: Author can switch the selected session without restarting the app, and every panel
  re-derives from the newly selected session. Priority: must-have. Change: new

### Model introspection

- FR-004: Author can view the trained model's architecture — layers, output shapes, parameter
  counts. Priority: must-have. Change: new
- FR-005: Author can view the session's feature contract as recorded in `manifest.json`: numeric
  columns, categorical columns, output column, and the post-one-hot `feature_names` order.
  Priority: must-have. Change: new
- FR-006: Author can view per-stage training history and the recorded test metrics without opening
  a CSV by hand. Priority: must-have. Change: new
- FR-007: Author can view the session's inventory — every artifact role and filename the manifest
  records, with presence confirmed on disk. Priority: nice-to-have. Change: new

### Global feature importance

- FR-008: Author can compute global permutation feature importance for the selected session against
  a dataset they choose. Priority: must-have. Change: new
- FR-009: Author sees importance reported per input column, with one-hot-expanded columns aggregated
  back to the originating categorical column. Priority: must-have. Change: new
- FR-010: Author sees progress while an importance run is in flight, and a completed run is reused
  rather than recomputed when the same session, dataset and seed are re-selected.
  Priority: must-have. Change: new
- FR-011: Importance computation is importable from the package independently of the dashboard, so
  it is not trapped in UI code. Priority: must-have. Change: new

### Single-row inference

- FR-012: Author can enter a value for each input column in a form and receive a prediction in
  engineering units without writing a file to disk. Priority: must-have. Change: new
- FR-013: The form's fields are generated from the selected session's feature contract, never from a
  hardcoded column list. Priority: must-have. Change: new

### Batch inference

- FR-014: Author can select a CSV or Parquet file and receive predictions for every row.
  Priority: must-have. Change: new
- FR-015: Author can download the batch predictions, with the option to include the input columns
  alongside the prediction column. Priority: must-have. Change: new
- FR-016: Author is told when input rows carried categorical levels the encoder never saw, including
  which column and which values. Priority: must-have. Change: new
- FR-017: `mlpp` can read a Parquet file wherever it currently reads a CSV for scoring input.
  Priority: must-have. Change: modified

### Explicitly preserved

- FR-018: `mlpp-train` and `mlpp-predict` continue to work with their current behavior and flags.
  Priority: must-have. Change: preserved
- FR-019: `config`, `data`, `preprocess`, `metrics`, `session` and `errors` continue to import no
  TensorFlow, and `uv run pytest -m 'not slow'` stays TF-free. Priority: must-have. Change: preserved
- FR-020: `session.py` remains the sole owner of every filename in a session directory; the
  dashboard resolves artifacts only through manifest roles. Priority: must-have. Change: preserved
- FR-021: `SCHEMA_VERSION` stays at 1 and `OUTPUTS/example/` continues to load unchanged.
  Priority: must-have. Change: preserved

- FR-022: Author is warned when a value entered in the single-row form falls outside the range the
  scaler observed during fitting. Priority: must-have. Change: new

### Socrates round (Phase 4.5)

Four FRs carried genuinely contested counter-arguments and were put to the author. The remainder are
recorded with reasoned resolutions derived from decisions already locked in Phases 1–3.

**Contested — resolved by the author:**

- FR-009 — *"Aggregation is a no-op: `categorical_columns` is empty in every session that exists."*
  Resolution: **kept as must-have.** The pipeline supports categoricals, and importance that reported
  a one-hot column instead of its originating column would be silently wrong the first time one is
  configured. Correctness over YAGNI; the cost is small.
- FR-010 — *"With 13 features and a small test file, permutation is seconds; caching and progress UI
  are premature."* Resolution: **kept as written, both parts.** Feature count and row count are both
  config-driven, so the cheap case is not the guaranteed case.
- FR-012 — *"One hand-typed row yields a confident number with no signal about whether the input is
  in-distribution."* Resolution: **kept, and hardened** — FR-022 added. The fitted scaler is already
  loaded, so an out-of-range check is nearly free and removes the footgun.
- FR-017 — *"`data.py` is a preserved TF-free module on the training path; a dashboard should not
  modify it."* Resolution: **kept, but constrained to additive.** A new suffix-dispatching reader is
  added alongside `read_csv_auto`, which is not itself changed. The training path keeps calling the
  existing function and therefore cannot regress, while Parquet becomes available to the whole
  package rather than to the UI alone.

**Uncontested — resolved from prior decisions:**

- FR-001/002/003 — *"Session selection is chrome; just take a path argument."* Resolution: kept.
  Eight session directories exist and three are broken; discovery plus a naming error message is the
  cheapest path to the Secondary success criterion.
- FR-004/005/006 — *"This duplicates the notebook report."* Resolution: kept. The notebook is a
  static demonstration bound to `OUTPUTS/example/`; these panels are live against any session.
- FR-007 — *"Inventory display is developer trivia."* Resolution: kept at **nice-to-have** — useful
  for diagnosing the broken sessions, not required for the Primary criterion.
- FR-008/011 — *"Importance belongs in the dashboard; a separate module is over-engineering."*
  Resolution: kept. Trapping it in UI code contradicts the Secondary criterion and makes it
  untestable in the fast suite.
- FR-013 — *"Hardcoding 13 fields is faster."* Resolution: kept. Hardcoding a column list is exactly
  the drift that the manifest-as-single-source-of-truth invariant exists to prevent.
- FR-014/015/016 — *"`mlpp-predict` already does batch."* Resolution: kept. It requires composing
  three paths in a terminal and cannot read Parquet; FR-016 surfaces `PredictionResult.unseen`, which
  the CLI reports only as a log line.
- FR-018..021 — preserved-behavior FRs. No counter-argument: these are the change's guardrails and
  were selected explicitly by the author in Phase 1.

## Business Logic Changes

**This change adds one new domain rule.** It is not, as originally scoped, purely a UI surface over
existing capability — three of the four surfaces are wrappers, but feature importance is a decision
the system makes, and nothing in `mlpp` makes it today. The Phase 1 Socrates question is resolved
here in favour of the wider scope.

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

The author encounters the rule as a ranked chart on the dashboard, after selecting a session and a
dataset. No other domain logic changes.

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
- The `dev` and `notebook` dependency groups; the new work lives in the `dashboard` group so that
  `uv sync` without it still yields a working train/predict install.

**Preserved behavior, explicitly.**
- The TF-free module set (`config`, `data`, `preprocess`, `metrics`, `session`, `errors`) imports no
  TensorFlow. The new dashboard subpackage sits deliberately on the TF-owning side of that line,
  alongside `model`, `training`, `plots`, `pipeline` and `predict`.
- `session.py` remains the sole owner of session filenames.
- Sessions are never written to. The dashboard has read-only access to `OUTPUTS/`.

## Non-Functional Requirements

- `uv run pytest -m 'not slow'` completes in roughly its current time (~1s) and imports no
  TensorFlow. Any test that requires the training stack carries the `slow` marker.
- Any operation taking longer than two seconds — model load, importance computation — shows
  continuous visible progress. The interface never appears frozen.
- No file inside a session directory is created, modified, or deleted by this change. Results the
  author wants to keep leave through the browser, not through `OUTPUTS/`.
- Every failure reaching the author names the artifact and the reason — a missing file, a schema
  version, an unparseable input — and never surfaces as a bare traceback. Errors originate from the
  `MlppError` hierarchy.
- `uv run mypy src/mlpp` stays clean under strict mode, and `ruff check` / `ruff format --check`
  stay clean, for the new subpackage as for the rest.
- The dashboard is reachable only from the machine it runs on.

## Product framing

- **Product type.** The existing system is a `data-pipeline` driven by CLI entrypoints. This change
  adds a local `web-app` surface alongside it; neither the pipeline nor the CLIs change type.
- **Target scale.** Unchanged: one user, one machine, one session at a time. `users: small`,
  negligible qps, dataset volume bounded by whatever the author can hold in a pandas frame.
- **Timeline.** `delivery_weeks: 3`, after hours, no hard deadline. Locked in Phase 3.

## Non-Goals

- **No training from the dashboard.** The UI reads sessions; it never launches a run. Training has
  long-running, checkpointing, sequential-fit semantics that belong to `mlpp-train` and would
  dominate the delivery budget if surfaced.
- **No SHAP, captum, or gradient attribution.** Permutation importance is the only method in scope.
  This also rules out per-row local explanations — the change delivers *global* importance only.
- **No hosting, deployment, or multi-user access.** No container image, no auth, no shared instance,
  no concurrency handling. Localhost only, run by hand.
- **No session mutation or cleanup tooling.** The dashboard will not delete, repair, rename, or
  regenerate sessions — including the three incomplete directories currently under `OUTPUTS/`. Those
  exist to be *reported* on, not fixed. Read-only, without exception.
- **No new artifact role and no `SCHEMA_VERSION` bump.** A non-goal because it was a live option:
  persisting an importance sample at train time was considered in Phase 4 and rejected precisely
  because it would breach this line.

## Forward: technical roadmap

Captured during discovery, outside the PRD schema. For downstream planning, not for `prd.md`.

- An uncommitted `dashboard` dependency group (`streamlit>=1.40,<2`) already sits in
  `apps/backend/pyproject.toml` with a resolved `uv.lock`. The implementing change should decide
  whether to keep, amend, or revert that working-tree state before building on it.
- `pyarrow` must be promoted from a transitive streamlit dependency to a declared direct one.
  Open question: whether it belongs in the base `dependencies` (Parquet becomes a core capability,
  since `data.py` gains the reader) or in the `dashboard` group.
- The dashboard subpackage lands at `apps/backend/src/mlpp/dashboard/` and sits on the
  TensorFlow-owning side of the module split. Whatever guards the TF-free rule today needs to be
  told about the new subpackage explicitly.
- `CLAUDE.md`'s "Python only — there is no frontend" statement and its layout section will both need
  updating; the run command (`streamlit run ...`) belongs in the Build & run block.
- Test placement follows the existing one-module-per-source-module convention; importance logic
  should be reachable in the fast suite with a stubbed scorer, keeping only genuine Keras paths
  behind the `slow` marker.

## User Stories

Derived from the FR set locked in Phase 4; delta-framed against today's CLI-only workflow.

### US-01: Author inspects a trained session without opening a file

- **Given** at least one session directory exists under `OUTPUTS/`
- **When** the author launches the dashboard and selects a session
- **Then** they see its architecture, its feature contract, its training history and its test metrics
  in one place

*Before:* the author opened `training_curves_*.png`, `test_metrics.csv` and `history_train_*.csv`
separately and reconstructed the picture by hand.

#### Acceptance criteria
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

#### Acceptance criteria
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

#### Acceptance criteria
- Form fields are generated from the session's feature contract.
- No file is written anywhere, including inside the session directory.

### US-04: Author scores a whole file

- **Given** a selected session
- **When** the author chooses a CSV or Parquet file
- **Then** they see predictions for every row and can download them, optionally with the input
  columns alongside

*Before:* CSV only, via the CLI, with no Parquet path.

#### Acceptance criteria
- Unrecognised categorical levels are reported in the interface, naming column and values.
- The download leaves through the browser; `OUTPUTS/` is not written to.

## Open Questions

1. **Does `pyarrow` belong in the base `dependencies` or in the `dashboard` group?** Since `data.py`
   gains the Parquet reader, Parquet arguably becomes a core capability that `mlpp-predict` should
   also get — which would put it in the base. Owner: author. Resolve during planning. Block: no.
2. **What happens to the uncommitted `dashboard` group and `uv.lock` changes already in the working
   tree?** Keep, amend, or revert before building. Owner: author. Resolve before implementation
   starts. Block: no.
3. **How many permutation repeats is the default?** Phase 5 settled on "several, seeded, mean and
   spread reported" but not the number. Owner: author. Resolve during planning. Block: no.
4. **Should `mlpp-predict` gain Parquet support in this change or later?** FR-017 puts the reader in
   `data.py`, which makes wiring it into the predict CLI nearly free — but that CLI is marked
   preserved. Owner: author. Resolve during planning. Block: no.

## Quality cross-check

Run at Phase 7 against the brownfield element set. All six present; status `accepted`.

| Element | Status |
| --- | --- |
| Access Control | present — localhost only, no auth; the new port-binding exposure is named |
| Business Logic | present — one declarative sentence; importance identified as a real domain rule |
| Project artifacts | present |
| Timeline-cost acknowledged | present — `delivery_weeks: 3`, within the gate |
| Non-Goals | present — five entries |
| Preserved behavior | present — `## Constraints & Compatibility` names all four preserved invariants |

No gaps to mirror into `/10x-prd`'s `## Open Questions`. The four entries under `## Open Questions`
above are genuine unknowns surfaced during discovery, not cross-check failures; none of them block.

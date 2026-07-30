# Notebook Consolidation Implementation Plan

## Overview

Close the verification gap that let `notebooks/01_train.ipynb` rot undetected, then
convert it from an empty-output training driver into a deterministic executive
demonstration report readable without running it.

## Current State Analysis

The frame brief established that two of the milestone's four stated objectives had
empty premises — there is exactly one notebook and it contains no extractable
business logic — and that the real problem is verification plus artifact role. This
plan takes that as settled.

What planning added: **executing the notebook, which nothing in this repo has ever
done, surfaced three further defects beyond the one already patched.**

| Finding | Evidence |
| --- | --- |
| Execution is cheap | All 9 cells in **18.0s**, including 5-epoch training on both TRAIN files (1500 rows each), 4 evaluations and plot generation. CI can afford an execution gate. |
| No new dependency needed | `nbclient` 0.11.0 and `nbconvert` 7.17.1 are already installed transitively via the `notebook` dep group's `jupyterlab`. |
| **Defect: IFrame crashes from `notebooks/`** | The last cell calls `.relative_to(Path.cwd())` on a path under `OUTPUTS/`. With cwd = `notebooks/` it raises `ValueError: … is not in the subpath of …/notebooks`. That is JupyterLab's cwd when the notebook is opened normally. Cell 1 handles `cwd=notebooks/` for `REPO_ROOT`; the IFrame cell then assumes repo root. |
| **Defect: `mlpp` kernel unregistered** | Executing as-committed raises `jupyter_client.kernelspec.NoSuchKernel: No such kernel named mlpp` unless the kernel is installed or the name overridden. |
| **Latent: no cell `id` fields** | `nbformat` emits `MissingIDFieldWarning: … this will become a hard error in future nbformat versions`. `build_notebook.py:132` never sets one. |
| Outputs land on 5 of 9 cells | `stream`, `execute_result`, `display_data`. `build_notebook.py:134` emits `"outputs": []` by construction, so the committed notebook shows a reader nothing. |
| The example is narratable | `OUTPUTS/example/test_metrics.csv` carries `train_file,test_file,mse,rmse,mae,r2` over 4 rows; `prediction_analysis_tr01_te01.html` and `manifest.json` are committed alongside. |

The already-patched defect (`21331ec`): cell 3 passed `DataConfig`'s pre-s-04 column
kwargs and raised `TypeError`. Diagnosed and fixed; the gap that allowed it is what
this plan closes.

## Desired End State

`notebooks/01_exploration_and_baseline.ipynb` is a committed, executed report that a
reader browsing GitHub sees rendered — feature contract, metrics table, live-scored
predictions, and an interactive truth-vs-prediction plot — all narrating the
committed `OUTPUTS/example/` session. It executes deterministically in ~seconds, a
test proves it executes, and the drift gate proves its source matches the generator.

Verify by: opening the committed `.ipynb` and seeing outputs without running
anything; `pytest -k notebook` passing; and the CI drift job passing against a
notebook that now carries outputs.

### Key Discoveries:

- `scripts/build_notebook.py:19-125` holds every cell as a string literal in `CELLS: list[tuple[str, str]]` — invisible to `mypy` and `ruff`.
- `.github/workflows/ci.yml:80-90` regenerates the JSON and diffs the whole file, validating formatting fidelity but never correctness.
- `build_notebook.py:134` sets `"outputs": []` and `"execution_count": None`; `:139-143` pins `kernelspec` to `mlpp`.
- IFrame `src` is resolved by the browser relative to the **notebook's own directory**, which is fixed — so a hardcoded `../OUTPUTS/example/...` needs no runtime path logic at all.
- Narrating the committed example makes the notebook deterministic, which removes the stale-output risk a source-only gate would otherwise carry.
- Rename blast radius is 14 references across 5 files; `frame.md` is one of them and must **not** be rewritten — it records what was true at framing time.
- `apps/backend/tests/` has no notebook test module; test paths are relative to `apps/backend/` (pytest configfile lives there), so a test reaching the notebook must go up two levels.

## What We're NOT Doing

- **No shared presentation module.** Deferred to s-06, which will extract helpers once the dashboard's real needs are known rather than guessing them now. The original "extract into `src/mlpp/`" objective stays a no-op in this change.
- **No second notebook.** Exactly one, per the milestone objective.
- **No training in the notebook.** `run_pipeline` leaves it; `mlpp-train` and the README already own that path and the pipeline is covered by tests.
- **No new dependencies.** `nbclient` is already present.
- **No change to `kernelspec`.** It stays `mlpp` for humans who install the kernel; the test overrides the name.
- **No `apps/ui`, no Streamlit/Taipy.** That is s-06 entirely.
- **No rewrite of `frame.md`** to reflect the patched breakage — it is a historical record.

## Implementation Approach

Three phases, ordered so the gate protects the rewrite rather than being retrofitted
onto it.

Phase 1 hardens the generator and the CI contract against the notebook *as it exists
today*, immediately closing the verification gap and fixing the two defects that
would otherwise make an execution gate fail. Phase 2 rewrites the content and renames
the artifact, now guarded by that gate. Phase 3 commits the executed outputs that
make it a report and trues up the docs.

## Critical Implementation Details

**Ordering.** The drift gate must become source-only *before* outputs are committed
(phase 3), or CI fails on the first executed commit. Phase 1 makes that change while
outputs are still empty, so the gate change is verifiable in isolation.

**The execution test asserts execution, not output equality.** Phase 1's test runs
against the still-nondeterministic training notebook, so comparing outputs would make
it flaky. Assert cells complete without raising; determinism arrives with phase 2 and
is a property of the content, not the gate.

**Kernel and cwd are both required arguments, not defaults.** `nbclient` will fail on
`kernel_name="mlpp"` (unregistered) and the notebook's own paths depend on cwd. The
test must pass an explicit kernel override and pin the working directory; leaving
either implicit reproduces one of the two defects found during planning.

## Phase 1: Harden the generator and the drift gate

### Overview

Make the notebook executable-and-verified: fix the cwd crash, emit cell ids, switch
the drift gate to comparing source only, and add a test that actually executes the
notebook. Content stays as-is.

### Changes Required:

#### 1. Cell identity in the generator

**File**: `scripts/build_notebook.py`

**Intent**: Emit a stable `id` per cell so `nbformat` stops warning and the notebook
survives the announced future hard error.

**Contract**: Each cell dict gains an `id` field, deterministic across runs (so
generation stays idempotent and the drift gate stays meaningful) and valid per
nbformat's schema (alphanumeric/`-`/`_`, 1–64 chars). Derive from cell index.

#### 2. Fix the IFrame path

**File**: `scripts/build_notebook.py`

**Intent**: Remove the `relative_to(Path.cwd())` call that crashes when the notebook
runs from its own directory.

**Contract**: The final code cell's `IFrame` source becomes a fixed
notebook-relative path string with no runtime path computation. The browser resolves
`src` relative to the notebook's directory, which is fixed, so no cwd logic is
needed. Phase 2 repoints it at the committed example.

#### 3. Drift gate compares source only

**File**: `.github/workflows/ci.yml`

**Intent**: Let the committed notebook carry outputs while still proving its source
matches the generator — the collision this plan exists to resolve.

**Contract**: The "Fail if it drifted" step stops using `git diff --quiet` on the
whole file. It compares only `cell_type` and `source` (plus cell `id`) of each cell,
plus notebook-level `metadata`, ignoring `outputs` and `execution_count` on both
sides. Implement as a small comparison invoked from the workflow rather than inline
shell, so it is runnable locally; the error message must keep naming the regeneration
command. Whether that comparison lives in `scripts/` as a sibling of
`build_notebook.py` is the implementer's call.

#### 4. Notebook execution test

**File**: `apps/backend/tests/test_notebook.py` (new)

**Intent**: Execute the committed notebook and fail if any cell raises — the gate
whose absence let two defects reach `main`.

**Contract**: Marked `@pytest.mark.slow` (it imports the training stack). Uses
`nbclient.NotebookClient` with an explicit kernel override (the committed
`kernelspec` names the unregistered `mlpp` kernel) and an explicitly pinned working
directory. Locates the notebook relative to the test file, not the cwd. A second test
asserts the notebook executes with cwd set to the notebook's own directory, pinning
the regression that the IFrame defect represented.

### Success Criteria:

#### Automated Verification:

- Generation is idempotent: running `scripts/build_notebook.py` twice yields identical files
- Notebook regeneration produces no `MissingIDFieldWarning`
- New tests pass: `cd apps/backend && uv run pytest tests/test_notebook.py -v`
- Full suite passes: `cd apps/backend && uv run pytest`
- Fast suite stays TensorFlow-free and fast: `uv run pytest -m 'not slow'`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- The source-only drift comparison passes against the committed notebook
- The drift comparison still *fails* when a cell's source is deliberately altered — proving it did not become a no-op

#### Manual Verification:

- Opening the notebook in JupyterLab from `notebooks/` and running all cells completes without error, including the final IFrame cell
- The drift job's failure message still tells a reader exactly how to regenerate

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to the next phase.

---

## Phase 2: Rewrite as a demonstration report

### Overview

Replace the training-driver content with a narrative that reads the committed
`OUTPUTS/example/` session, and rename the artifact to match its new role.

### Changes Required:

#### 1. Report content

**File**: `scripts/build_notebook.py`

**Intent**: Rewrite `CELLS` as an executive report: what the project is, what the
model expects, how it performed, and what its predictions look like — narrating the
committed reference session rather than training a new one.

**Contract**: The cell sequence covers, with markdown framing each step: the feature
contract restored from the session manifest via `load_session` (numeric/categorical
split, expanded feature count); the recorded test metrics from the session's
`test_metrics.csv`; predictions produced through the s-04 scoring core
(`mlpp.predict.load_model` + `score_frame`) against a `TEST/*.csv`, shown in
engineering units next to the actual target; and the committed interactive
truth-vs-prediction HTML via `IFrame`. No `run_pipeline` call. Prose points a reader
at `mlpp-train` / `mlpp-predict` for running things themselves. Filenames come from
the manifest or the documented `OUTPUTS/example/` layout — the notebook must not
invent session filenames.

#### 2. Rename the artifact

**File**: `scripts/build_notebook.py`, `.github/workflows/ci.yml`, `README.md`, `CLAUDE.md`

**Intent**: `01_train` misdescribes a notebook that no longer trains; rename to
`01_exploration_and_baseline.ipynb` and update every live reference.

**Contract**: `build_notebook.py`'s `NOTEBOOK` target and module docstring; the three
`ci.yml` references in the drift job (path, diff target, error message); README's
structure tree, the `build_notebook.py` description, the notebook-open instruction
and the CI section; CLAUDE.md's layout bullet and notebook instruction. The old file
is deleted, not left behind. `context/changes/s-05-notebook-consolidation/frame.md`
is **excluded** — it records the pre-change state.

#### 3. Determinism check

**File**: `apps/backend/tests/test_notebook.py`

**Intent**: Assert the report is reproducible, which is what makes committed outputs
trustworthy and git diffs clean.

**Contract**: A slow-marked test executes the notebook twice and asserts the
text-bearing outputs match. Scope it to deterministic output types — a matplotlib
`display_data` payload or an HTML repr containing a session timestamp is not
guaranteed byte-stable, so compare `stream` and `execute_result` text and state the
exclusion in the test's docstring.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- Notebook tests pass, including the determinism check: `uv run pytest tests/test_notebook.py -v`
- No reference to `01_train` survives outside `frame.md`: `grep -rn "01_train" --include="*.py" --include="*.yml" --include="*.md" .`
- The old `notebooks/01_train.ipynb` no longer exists
- The notebook contains no `run_pipeline` call
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- Source-only drift comparison passes for the renamed notebook

#### Manual Verification:

- The narrative reads as an executive report — a recruiter skimming it learns what the model does and how well, without running anything
- Metrics shown match `OUTPUTS/example/test_metrics.csv`
- The predictions cell shows values in the target's real units, not scaled space
- The IFrame renders the committed interactive plot when opened from `notebooks/`

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding.

---

## Phase 3: Commit executed outputs and document the report

### Overview

Store the executed outputs so the committed notebook renders on GitHub, and update
the docs to describe what the notebook now is.

### Changes Required:

#### 1. Executed notebook committed with outputs

**File**: `notebooks/01_exploration_and_baseline.ipynb`

**Intent**: A reader must see results without running anything — the whole point of
the demo-report role.

**Contract**: The committed `.ipynb` carries outputs and execution counts from a
clean run against `OUTPUTS/example/`. How outputs get there — a documented manual
regenerate-then-execute step, or a flag on the generator — is the implementer's call,
but the chosen route must be written down wherever the regeneration command already
appears, because a future contributor running only `build_notebook.py` will otherwise
strip the outputs and see a confusing diff.

#### 2. Documentation

**File**: `README.md`, `CLAUDE.md`

**Intent**: Describe the notebook as a demonstration report over the committed
example, and record how to regenerate it now that outputs are part of the artifact.

**Contract**: README's notebook section explains it narrates `OUTPUTS/example/`
rather than training, that it needs no data or GPU to read, and gives the
regenerate-with-outputs procedure. CLAUDE.md's notebook bullet replaces "thin driver"
with the report role and keeps the never-hand-edit invariant, noting that outputs are
committed and the drift gate compares source only.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- The committed notebook carries outputs on its code cells (not `"outputs": []`)
- Source-only drift comparison passes against the committed notebook *with* outputs — the collision is resolved
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- CI passes on the branch (both Python versions plus the drift job)

#### Manual Verification:

- The notebook renders with visible outputs on GitHub's blob view
- Committed notebook JSON has not grown unreasonably (no unexpected embedded binary payloads)
- Following README's regeneration procedure reproduces the committed notebook, outputs included
- CLAUDE.md would stop a future agent from hand-editing the notebook or committing it output-stripped

**Implementation Note**: After completing this phase and all automated verification passes, pause for manual confirmation from the human.

## Testing Strategy

### Unit Tests:

- The source-only drift comparison detects an altered cell source (guards against the check becoming a no-op)
- The comparison ignores `outputs` and `execution_count` differences

### Integration Tests:

- Notebook executes end to end without raising (the gate whose absence caused this change)
- Notebook executes with cwd set to `notebooks/` — regression pin for the IFrame defect
- Notebook executes deterministically across two runs, for text-bearing outputs

### Manual Testing Steps:

1. Open the committed `.ipynb` on GitHub and confirm outputs render without running
2. Open it in JupyterLab from `notebooks/`, Run All, confirm no cell errors and the IFrame renders
3. Compare the metrics table against `OUTPUTS/example/test_metrics.csv`
4. Confirm predictions are in the target's real units
5. Follow README's regeneration procedure and confirm the result matches what is committed
6. Deliberately edit a cell's source in the `.ipynb`, run the drift comparison, and confirm it fails

## Performance Considerations

Narrating the committed example replaces an 18s training run with reading a session
plus one forward pass, so the notebook itself becomes seconds rather than tens of
seconds. The notebook test imports TensorFlow (via the scoring core), so it is
slow-marked and must not leak into the fast suite — `pytest -m 'not slow'` staying
TensorFlow-free is an explicit criterion in every phase. Committed outputs add JSON
weight; because the report shows tables and an IFrame rather than inline matplotlib
figures, no large base64 payloads are expected, and phase 3 verifies that manually.

## Migration Notes

The rename deletes `notebooks/01_train.ipynb` and adds
`notebooks/01_exploration_and_baseline.ipynb`; anyone with a local checkout gets a
delete plus an add rather than a rename-in-place, and a stale kernel selection may
need re-picking. The drift gate's semantics change from whole-file to source-only, so
a contributor who previously relied on it to catch output churn no longer will —
called out in CLAUDE.md by phase 3.

## References

- Frame brief: `context/changes/s-05-notebook-consolidation/frame.md`
- Patched breakage: commit `21331ec` (config cell vs `ColumnConfig`)
- Generator: `scripts/build_notebook.py:17` (target), `:19-125` (cells as string literals), `:132` (cell dict, no id), `:134` (empty outputs), `:139-143` (kernelspec `mlpp`)
- Drift job: `.github/workflows/ci.yml:80-90`
- Scoring core this report will use: `apps/backend/src/mlpp/predict.py`, `apps/backend/src/mlpp/session.py:310` (`load_session`)
- Committed session narrated: `OUTPUTS/example/manifest.json`, `test_metrics.csv`, `prediction_analysis_tr01_te01.html`
- Prior change: `context/archive/2026-07-30-s-04-production-inference-seam/plan.md`

## Progress

> Convention: `- [ ]` pending, `- [x]` done. Append ` — <commit sha>` when a step lands. Do not rename step titles. See `references/progress-format.md`.

### Phase 1: Harden the generator and the drift gate

#### Automated

- [x] 1.1 Generation is idempotent: running `scripts/build_notebook.py` twice yields identical files — d700f3d
- [x] 1.2 Notebook regeneration produces no `MissingIDFieldWarning` — d700f3d
- [x] 1.3 New tests pass: `cd apps/backend && uv run pytest tests/test_notebook.py -v` — d700f3d
- [x] 1.4 Full suite passes: `cd apps/backend && uv run pytest` — d700f3d
- [x] 1.5 Fast suite stays TensorFlow-free and fast: `uv run pytest -m 'not slow'` — d700f3d
- [x] 1.6 Type checking passes: `uv run mypy src/mlpp` — d700f3d
- [x] 1.7 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — d700f3d
- [x] 1.8 The source-only drift comparison passes against the committed notebook — d700f3d
- [x] 1.9 The drift comparison still fails when a cell's source is deliberately altered — d700f3d

#### Manual

- [x] 1.10 Running all cells in JupyterLab from `notebooks/` completes without error, including the final IFrame cell — d700f3d
- [x] 1.11 The drift job's failure message still tells a reader how to regenerate — d700f3d

### Phase 2: Rewrite as a demonstration report

#### Automated

- [x] 2.1 Full suite passes: `cd apps/backend && uv run pytest` — d86a6b7
- [x] 2.2 Notebook tests pass, including the determinism check: `uv run pytest tests/test_notebook.py -v` — d86a6b7
- [x] 2.3 No reference to `01_train` survives outside `frame.md` — d86a6b7
- [x] 2.4 The old `notebooks/01_train.ipynb` no longer exists — d86a6b7
- [x] 2.5 The notebook contains no `run_pipeline` call — d86a6b7
- [x] 2.6 Type checking passes: `uv run mypy src/mlpp` — d86a6b7
- [x] 2.7 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — d86a6b7
- [x] 2.8 Source-only drift comparison passes for the renamed notebook — d86a6b7

#### Manual

- [x] 2.9 The narrative reads as an executive report — understandable without running anything — d86a6b7
- [x] 2.10 Metrics shown match `OUTPUTS/example/test_metrics.csv` — d86a6b7
- [x] 2.11 The predictions cell shows values in the target's real units, not scaled space — d86a6b7
- [x] 2.12 The IFrame renders the committed interactive plot when opened from `notebooks/` — d86a6b7

### Phase 3: Commit executed outputs and document the report

#### Automated

- [x] 3.1 Full suite passes: `cd apps/backend && uv run pytest` — ca1630d
- [x] 3.2 The committed notebook carries outputs on its code cells (not `"outputs": []`) — ca1630d
- [x] 3.3 Source-only drift comparison passes against the committed notebook with outputs — ca1630d
- [x] 3.4 Type checking passes: `uv run mypy src/mlpp` — ca1630d
- [x] 3.5 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — ca1630d
- [x] 3.6 CI passes on the branch (both Python versions plus the drift job) — ca1630d

#### Manual

- [x] 3.7 The notebook renders with visible outputs on GitHub's blob view — ca1630d
- [x] 3.8 Committed notebook JSON has not grown unreasonably — ca1630d
- [x] 3.9 Following README's regeneration procedure reproduces the committed notebook, outputs included — ca1630d
- [x] 3.10 CLAUDE.md would stop a future agent from hand-editing or output-stripping the notebook — ca1630d

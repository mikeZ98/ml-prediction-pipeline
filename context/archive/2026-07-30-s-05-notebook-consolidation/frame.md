# Frame Brief: Notebook consolidation & core engine consolidation

> Framing step before /10x-plan. This document captures what is *actually*
> at issue, separated from what was initially assumed.

## Reported Observation

The repository is being polished for top-tier recruiter visibility. As the first
of two milestones (s-05, then s-06's `apps/ui` dashboard), the notebook layer
should be consolidated: audit every `.ipynb`, remove stale/duplicate/draft ones,
retain exactly ONE production-grade notebook serving as an executive
demonstration report; extract any remaining data prep, time-series
transformations, feature engineering or plot utilities into typed, tested modules
under `src/mlpp/`; rewrite the notebook to import from those modules; keep
pytest, `mypy --strict`, ruff and the notebook-drift CI job green.

## Initial Framing (preserved)

- **User's stated cause or approach**: multiple `.ipynb` files exist — some stale, duplicate or draft — and the surviving notebook still defines business logic (data prep, transforms, feature engineering, plot utilities) inside its cells.
- **User's proposed direction**: delete the extras, extract the logic into `src/mlpp/`, rewrite the notebook to import it, hold all four quality gates green.
- **Pre-dispatch narrowing**: (1) *"Both breakage and presentation"* — the live breakage and the presentation quality are two distinct observations to be addressed together in one change. (2) *"Executive demo report"* — the notebook should become a narrative walkthrough readable **without running it** (data → model → metrics → truth-vs-prediction, with outputs visible), which implies the rename and stored cell outputs.

## Dimension Map

The observation could originate at any of these dimensions:

1. **Notebook inventory** — stale/duplicate/draft `.ipynb` files accumulating in the repo. ← initial framing
2. **Logic resident in notebook cells** — business logic defined in cells rather than imported. ← initial framing
3. **Verification gap** — nothing executes or type-checks the notebook, so it can rot silently.
4. **Generation architecture** — the notebook is a generated build artifact whose source is string literals, making "rewrite the notebook" category-mismatched.
5. **Artifact role mismatch** — what exists is a thin training driver; what the goal needs is an executive demonstration report.

## Hypothesis Investigation

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| **1. Notebook inventory needs pruning** | `find . -name "*.ipynb"` excluding `.venv`/`.uv_cache` returns exactly one path: `notebooks/01_train.ipynb`. No `.ipynb_checkpoints` anywhere. There is nothing stale, duplicate or draft to remove. | **NONE** |
| **2. Logic lives in the cells** | All five code cells total ~1.8 KB and contain only: imports, three dataclass constructions, `run_pipeline(cfg)`, `pd.DataFrame([row.as_dict() ...])`, and an `IFrame` embed. No data prep, no transform, no feature engineering, no plot utility is defined anywhere in the notebook. Commit `8a05ce7` ("refactor: extract notebook pipeline into tested mlpp package") already performed this extraction; `CLAUDE.md:15` records the resulting invariant ("thin driver over `mlpp`; **never** re-add pipeline logic to it"). | **NONE** |
| **3. Verification gap** | `scripts/build_notebook.py:19-125` holds every cell as a **string literal** in `CELLS: list[tuple[str, str]]`, so `mypy` and `ruff` see strings, not code. No test imports or executes the notebook. CI's only notebook job (`.github/workflows/ci.yml:80-90`) regenerates the JSON and diffs it — validating *formatting fidelity*, never *correctness*. **Consequence, verified:** cell 3 passes `input_columns=`/`output_column=`/`categorical_columns=`/`strict_schema=` directly to `DataConfig(...)`, but s-04 phase 1 (`4c2411a`) moved those into `ColumnConfig`. Executing the notebook today raises `TypeError: DataConfig.__init__() got an unexpected keyword argument 'input_columns'`. The drift job ran green during s-04 phase 3 while the notebook was already broken. | **STRONG** |
| **4. Generation architecture** | `scripts/build_notebook.py:17` writes `notebooks/01_train.ipynb`; `ci.yml:86` fails the build if the committed file differs from a fresh regeneration. The `.ipynb` is therefore a build artifact and hand-editing it breaks CI — the editable unit is `build_notebook.py`. Objective 3 ("rewrite the primary notebook") is not performable as literally stated. | **STRONG** |
| **5. Artifact role mismatch** | The notebook is titled "Training pipeline" and described as a "thin driver"; it trains live via `run_pipeline(cfg)` and embeds a *relative* IFrame to a freshly produced session. `build_notebook.py:134` emits `"outputs": []` and `"execution_count": None`, so the committed notebook shows **nothing** to a reader who does not run it. Confirmed by the user's narrowing answer: the intended artifact is a demo report readable without execution. | **STRONG** |

## Narrowing Signals

- **"Both breakage and presentation."** Rules out treating this as a single-axis cleanup. Two distinct problems ride in one change, and the plan must address both rather than collapsing one into the other.
- **"Executive demo report."** Decisive on dimension 5, and it surfaces a hard architectural collision (below). A reader-without-running artifact needs stored outputs; the current generator emits empty ones by construction.
- **The audit itself was decisive on dimensions 1 and 2** — one notebook, zero extractable logic. Two of four stated objectives have empty premises.
- **`OUTPUTS/example/prediction_analysis_tr01_te01.html` exists and is committed.** A demo report could narrate the *committed reference run* instead of training live — deterministic, fast, and already exercised by s-04's `test_committed_reference_run_is_scorable`.

## Cross-System Convention

This project has already been bitten by exactly this class of defect, and its
established convention is the fix. s-03's frame brief recorded: *"The committed
reference run `OUTPUTS/example/` is already rejected by the current loader …
nothing detects that but a runtime failure."* s-03 closed that by adding a
load-and-verify path plus a test that exercises the committed artifact; s-04
extended it with `test_committed_reference_run_is_scorable`. The convention is
therefore explicit in this repo: **an artifact is not trustworthy until something
executes it in CI.** Sessions got that treatment; the scoring path got it; the
notebook is the last artifact that never runs. The leading hypothesis matches the
convention exactly — it is the same defect, one layer up.

## Reframed (or Confirmed) Problem Statement

> **The actual problem to plan around is**: the notebook is the repository's last
> unverified artifact — generated from string literals that no type checker or
> test ever executes — and it is already broken on `main`; separately, it is the
> wrong *kind* of artifact for the goal, being an empty-output training driver
> where an executive demonstration report is wanted.

Two of the four stated objectives have empty premises: there is exactly one
notebook (nothing to audit away) and it contains no business logic (nothing to
extract). Objective 3 is already satisfied in substance and impossible as
literally worded, because the notebook is a generated artifact. What survives
from the original framing is objective 4 — the quality gates — and it survives
*inverted*: the gates are not a constraint to preserve, they are the defect. The
drift gate's green light is precisely what let the breakage through, so the plan's
core job is to make the notebook executed rather than merely diffed.

If addressed: the notebook stops being able to rot silently, the next config
change cannot break it undetected, and a recruiter skimming the repo sees a
narrative report with real metrics and charts instead of an unexecuted driver.

**Two collisions the plan must resolve rather than inherit:**

1. **Stored outputs vs. the drift gate.** An executive report needs visible outputs; `build_notebook.py:134` emits `"outputs": []` by construction, and `ci.yml:86` fails on any diff from regeneration. Committing an executed notebook breaks the gate as currently written. Whether the gate compares source-only, whether CI executes the notebook, or whether generation changes shape is a design decision.
2. **Extraction vs. creation.** There is nothing to *extract* — but a demo report and s-06's `apps/ui` dashboard will both want the same presentation helpers (manifest metrics, truth-vs-prediction rendering) over `OUTPUTS/` and `LoadedSession`. Objective 2 is a no-op as *extraction* and may be real as *creation of shared, typed, tested presentation code*. Whether that belongs in s-05 or waits for s-06 to reveal its real shape is a scoping decision, and building it speculatively risks guessing the dashboard's needs wrong.

## Confidence

**HIGH** — a verified runtime failure reproduced against the installed package,
a mechanical explanation for why no gate caught it (`build_notebook.py:19-125`
string literals, `ci.yml:80-90` diff-only), two of the stated premises falsified
by direct inventory, a decisive user narrowing signal on the artifact's role, and
a cross-system check that found the same defect class already recorded and fixed
twice in this repo's own history.

## What Changes for /10x-plan

Plan around **verification and role**, not consolidation: there is no notebook to
delete and no logic to extract. The plan should fix the live `ColumnConfig`
breakage, close the gap that let it through (make CI execute the notebook, not
just diff it), and convert the artifact from an empty-output training driver into
a readable demonstration report — resolving the stored-outputs-vs-drift-gate
collision explicitly. Treat the shared presentation helpers as an open scoping
question against s-06 rather than a settled deliverable.

## References

- Source files: `notebooks/01_train.ipynb` (the only notebook), `scripts/build_notebook.py:17` (generation target), `:19-125` (cells as string literals), `:134` (empty outputs by construction), `.github/workflows/ci.yml:80-90` (drift job, diff-only), `apps/backend/src/mlpp/config.py:34-67` (`ColumnConfig`/`DataConfig` split that broke the notebook), `CLAUDE.md:15` (thin-driver invariant), `README.md:24,31,95,176` (notebook references that a rename would touch)
- Verified failure: `DataConfig(..., input_columns=...)` → `TypeError: DataConfig.__init__() got an unexpected keyword argument 'input_columns'`
- Prior occurrences of this defect class: `context/archive/2026-07-29-s-03-algorithm-and-architecture-deep-optimization/frame.md:42`, and s-04's `test_committed_reference_run_is_scorable`
- Investigation: direct read of the notebook, its generator, the CI workflow and the archived frames; no sub-agents dispatched (single notebook, 154-line generator, recently-worked codebase)

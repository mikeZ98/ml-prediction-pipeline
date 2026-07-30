# Notebook Consolidation — Plan Brief

> Full plan: `context/changes/s-05-notebook-consolidation/plan.md`
> Frame brief: `context/changes/s-05-notebook-consolidation/frame.md`

## What & Why

The notebook is the repository's last unverified artifact — generated from string
literals that no type checker or test ever executes — and it was already broken on
`main`; separately, it is the wrong *kind* of artifact for the goal, being an
empty-output training driver where an executive demonstration report is wanted. This
plan closes the verification gap and changes the artifact's role.

## Starting Point

Exactly one notebook exists and it contains no extractable business logic, so two of
the milestone's four stated objectives had empty premises. The live breakage (s-04
moved `DataConfig`'s column fields into `ColumnConfig`; the config cell still used the
old signature) is diagnosed and patched in `21331ec`. What remains is the gap that
allowed it: `build_notebook.py` holds cells as string literals invisible to `mypy` and
`ruff`, and CI's only notebook job diffs regenerated JSON — validating formatting
fidelity, never correctness.

## Desired End State

`notebooks/01_exploration_and_baseline.ipynb` is a committed, executed report that
renders on GitHub without anyone running it: feature contract from the session
manifest, recorded metrics, live-scored predictions in engineering units, and the
interactive truth-vs-prediction plot — all narrating the committed `OUTPUTS/example/`
session. A test proves it executes; the drift gate proves its source matches the
generator.

## Key Decisions Made

| Decision | Choice | Why (1 sentence) | Source |
| --- | --- | --- | --- |
| What the real problem is | Verification + artifact role, not consolidation | One notebook, zero extractable logic — the stated cleanup had nothing to act on. | Frame |
| Artifact role | Executive demo report | Must be readable without running it, which the empty-output driver never was. | Frame |
| Outputs vs drift gate | Gate compares source only | Lets the committed notebook carry outputs while the generator stays the single source of truth for source. | Plan |
| Data source | Narrate committed `OUTPUTS/example/` | Deterministic and seconds-fast, and it exercises the s-04 scoring core the s-06 dashboard will also use. | Plan |
| Execution gate | pytest test using `nbclient` | Runs locally and in CI with zero new dependencies — `nbclient` is already installed via jupyterlab. | Plan |
| IFrame path | Fixed notebook-relative string | The notebook's own directory is fixed and the browser resolves `src` against it, so no runtime path logic can break. | Plan |
| Shared presentation helpers | Defer to s-06 | Avoids designing typed, tested code against a guess at the dashboard's needs. | Plan |
| Filename | Rename to `01_exploration_and_baseline` | `01_train` misdescribes a notebook that no longer trains. | Plan |

## Scope

**In scope:**
- Cell `id` fields and the IFrame cwd fix in the generator
- Drift gate switched from whole-file diff to source-only comparison
- New `tests/test_notebook.py`: executes the notebook, pins the cwd regression, checks determinism
- Content rewritten as a report narrating `OUTPUTS/example/` via `load_session` + the s-04 scoring core
- Rename and all live reference updates (README, CLAUDE.md, `ci.yml`, `build_notebook.py`)
- Executed outputs committed, plus docs for the new regeneration procedure

**Out of scope:**
- Shared presentation module (→ s-06)
- `apps/ui`, Streamlit/Taipy (→ s-06 entirely)
- Training inside the notebook — `mlpp-train` owns that path
- A second notebook; new dependencies; `kernelspec` changes
- Rewriting `frame.md` to reflect the patch — it is a historical record

## Architecture / Approach

```
build_notebook.py  ──generates──>  01_exploration_and_baseline.ipynb
       │                                    │
       │                          ┌─────────┴──────────┐
       │                          │                    │
  source-only diff  <─────────────┘            nbclient execution
  (CI: source matches                          (pytest: it actually runs)
   generator, outputs ignored)                          │
                                                        v
                              reads OUTPUTS/example/ via load_session
                              + mlpp.predict (s-04 scoring core)
```

Two independent gates replace one weak gate: the drift comparison proves the notebook
was not hand-edited, and the execution test proves it works. Neither alone would have
caught the breakage that motivated this change.

## Phases at a Glance

| Phase | What it delivers | Key risk |
| --- | --- | --- |
| 1. Harden generator + gate | Cell ids, IFrame fix, source-only drift comparison, execution test | The source-only comparison silently becoming a no-op — mitigated by a criterion that it must still *fail* on an altered cell |
| 2. Rewrite as demo report | Report narrating the committed example; rename + 14 reference updates | A missed reference; nothing catches one automatically because the gate only knows the path it is told |
| 3. Commit outputs + docs | Rendered notebook on GitHub, regeneration procedure documented | A future contributor running only `build_notebook.py` strips the outputs and sees a confusing diff |

**Prerequisites:** s-04 archived and pushed; `21331ec` patch on `main`; `OUTPUTS/example/` committed and loadable.
**Estimated effort:** ~2 sessions across 3 phases; phase 2 is the largest diff, phase 1 the highest value.

## Open Risks & Assumptions

- **Committed outputs can go stale silently.** The source-only gate deliberately ignores outputs, so nothing detects outputs that no longer match the code. Narrating the deterministic committed example rather than a live training run is what keeps this manageable; a stale-output check is not in scope.
- **The regeneration procedure becomes a two-step ritual.** `build_notebook.py` alone produces an output-free notebook, so anyone regenerating without executing will appear to delete all outputs. Phase 3 documents this wherever the command already appears, but the trap is real.
- **Determinism is assumed, not proven, for non-text outputs.** The determinism test deliberately scopes to `stream` and `execute_result` text; HTML reprs containing timestamps or matplotlib payloads are excluded and could still churn.
- **The notebook stops demonstrating training.** That capability moves entirely to `mlpp-train` and the test suite. If a reader expects the notebook to prove the pipeline trains, it no longer does.

## Success Criteria (Summary)

- A recruiter opening the notebook on GitHub sees the feature contract, real metrics and an interactive plot without running anything.
- The next API change that breaks the notebook fails a test instead of reaching `main` unnoticed.
- Fast suite stays TensorFlow-free and ~3s; full suite, `mypy --strict` and `ruff` stay clean, and CI's drift job passes against a notebook that carries outputs.

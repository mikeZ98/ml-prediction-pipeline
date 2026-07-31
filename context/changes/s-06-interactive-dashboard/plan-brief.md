# Interactive Dashboard — Plan Brief

> Full plan: `context/changes/s-06-interactive-dashboard/plan.md`
> PRD: `context/foundation/prd.md`
> Stack assessment: `context/foundation/stack-assessment.md`
> Health check: `context/foundation/health-check.md`

## What & Why

Inspecting a trained model is currently a file-archaeology exercise: opening a directory of CSVs,
PNGs and standalone HTML reports and reconstructing the picture by hand. This change adds a local
browser dashboard delivering four surfaces — session introspection, global feature importance,
single-row inference, and batch CSV/Parquet inference. Three are presentation layers over existing
capability; feature importance is genuinely new domain logic and does not exist in `mlpp` today.

## Starting Point

An `mlpp` uv project under `apps/backend/` with two console entrypoints and no interface of any
kind. Eight session directories exist under `OUTPUTS/`, three of them incomplete. The package is
deliberately split into a TensorFlow-free half and a TensorFlow-owning half — a split verified to
hold in fact, not just intent: running the 98-test fast suite leaves neither `tensorflow` nor
`keras` in `sys.modules`. Two blockers sit outside the code: both instruction files currently
forbid a frontend, and a `joblib`/`numpy` deprecation fires on the artifact-loading path every
panel will depend on.

## Desired End State

One command starts a local dashboard. The author picks any session under `OUTPUTS/` and — without
returning to a terminal — reads its architecture and feature contract, computes a ranked
feature-importance chart against a dataset they choose, scores a hand-entered row with an
out-of-range warning, and scores a whole CSV or Parquet file with a download. An invalid session
produces a message naming the problem rather than a traceback.

## Key Decisions Made

| Decision | Choice | Why (1 sentence) | Source |
| --- | --- | --- | --- |
| Importance method | Permutation only, R² degradation | Fits the three-week budget and stays comparable across datasets with different target scales. | PRD |
| Importance data source | Author picks a file at run time | A session stores no data, and persisting a sample would need a new artifact role and a schema bump. | PRD |
| Permutation target | Shuffle the raw column, not `X` | One-hot expansion happens after `align_columns`, so shuffling the source column groups its one-hot positions for free. | Plan |
| Importance module boundary | Parameterised over a scorer callable | Keeps the module TensorFlow-free, so it lands in the fast suite and satisfies FR-011 and FR-019 together. | Plan |
| Repeats default | 5, seeded, configurable | Enough to expose whether a ranking gap is noise; ~66 passes at current feature count is seconds. | Plan |
| Parquet reader shape | New `read_table_auto` sibling | `read_csv_auto` is untouched, so the training path cannot regress by construction. | PRD |
| `pyarrow` placement | Base dependencies | `read_table_auto` lives in a core module; an optional dependency would make it a latent `ImportError`. | Plan |
| `mlpp-predict` Parquet | Yes, swap its reader | One line once the reader exists; makes Parquet a package capability rather than UI-only. | Plan |
| Instruction files | Fixed in Phase 1 | Every later phase then runs against instructions that permit and describe the work. | Plan |
| joblib deprecation | Tripwire now, fix separately | Converts a future silent break into a failing test without pulling a dependency investigation into a UI plan. | Health check |
| Code placement | `src/mlpp/dashboard/` | One uv project, one lockfile, one test suite; explicitly TensorFlow-owning. | Stack assessment |

## Scope

**In scope:** session discovery and selection with named errors; architecture, contract, history,
metrics and inventory panels; permutation feature importance as a library module plus its panel;
single-row inference with an out-of-range warning; batch CSV/Parquet inference with download and
unseen-level reporting; `read_table_auto` and wiring `mlpp-predict` to it; instruction-file
corrections and the joblib tripwire.

**Out of scope:** training from the dashboard; SHAP, captum, or any gradient attribution;
per-row local explanations; hosting, containers, authentication, multi-user; session mutation or
cleanup; any change to `read_csv_auto` or the training path; the underlying joblib/numpy version
fix; the health check's `.editorconfig`, `py.typed` and Dependabot items.

## Architecture / Approach

`mlpp.dashboard` renders; it does not compute. A single `app.py` owns page config, the sidebar
session selector, and tab dispatch, wrapping every panel so `MlppError` surfaces as a readable
message. A `loaders.py` module centralises every artifact read and every cache decision —
`@st.cache_resource` for the unhashable Keras model, `@st.cache_data` for frames and importance
results keyed on `(session_dir, dataset_path, seed, n_repeats)`. Panels receive already-loaded
objects, so no panel ever touches a filename; artifacts resolve only through manifest roles.
Feature importance lives in `mlpp/importance.py` as a normal library module, parameterised over a
`Scorer` callable that `predict.make_scorer` supplies — which is what keeps TensorFlow out of it.

## Phases at a Glance

| Phase | What it delivers | Key risk |
| --- | --- | --- |
| 1. Prerequisites & contract | Instruction files unblocked, deps declared, joblib tripwire armed | The tripwire fails immediately on 30 known warnings; the scoped ignore must not silence future ones |
| 2. Parquet reader | `read_table_auto` + `mlpp-predict` wired to it | Touching a CLI marked preserved — needs an explicit CSV-unchanged test |
| 3. Feature importance | `mlpp/importance.py`, TF-free, fast-suite tested | Cumulative-permutation bug looks plausible while being wrong; API signature is what Phases 4–5 depend on |
| 4. Dashboard shell & introspection | App, session selection, error surfacing, four read-only panels | Streamlit's rerun model makes uncached loading silently expensive |
| 5. Inference panels | Importance chart, single-row form, batch inference | Range-check thresholds and download shape are the only new logic; rest is assembly |

**Prerequisites:** the uncommitted `dashboard` dependency group and `uv.lock` changes stay in the
working tree and are built on in Phase 1. No external access or prior work needed.

**Estimated effort:** ~3 weeks of after-hours work across 5 phases, matching the PRD's
`delivery_weeks: 3`. Phases 1–2 are short; Phase 3 carries the design risk; Phases 4–5 are the bulk
of the surface area but the least uncertainty.

## Open Risks & Assumptions

- The `joblib`/`numpy` deprecation is deferred, not fixed. `numpy>=2.0,<3` permits the version that
  removes it, so a lockfile refresh could turn the tripwire red at a time not of your choosing —
  which is the intended failure mode, but it will interrupt.
- Streamlit contributes no layout conventions, so Phase 1's conventions block is doing real
  structural work rather than documenting after the fact. If it is written loosely, Phases 4–5 drift.
- Importance cost scales linearly in both column count and repeats. Current numbers are comfortable;
  a much wider feature configuration would make the cache load-bearing rather than a convenience.
- Assumes a browser-renderable chart with error bars is achievable without adding a plotting
  dependency beyond the `plotly` already present.

## Success Criteria (Summary)

- From one command, all four surfaces are reachable without returning to a terminal, and every
  artifact read resolves through a manifest role rather than a hardcoded filename.
- The fast suite stays TensorFlow-free and near its 1.56s baseline, with importance logic covered
  in it via a stub scorer.
- Selecting one of the three incomplete session directories produces a message naming what is
  missing; `OUTPUTS/` is byte-unchanged after a full session.

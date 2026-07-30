# Frame Brief: Next epic after s-03 — inference seam vs. architectures vs. dashboard

> Framing step before /10x-plan. This document captures what is *actually*
> at issue, separated from what was initially assumed.

## Reported Observation

s-03 landed a validated single-owner artifact contract (`session.py` +
`manifest.json`, TF-free data/session/preprocess layers, 119 tests green, strict
mypy clean). Three candidate epics are on the table and one must be selected as
the next change:

- **A** — Production inference seam: `mlpp-predict` CLI + lightweight FastAPI service (`apps/api`) on `load_session()`.
- **B** — Next-gen ML baselines: LightGBM/CatBoost tabular baseline, State Space Models / Mamba-Tab, zero-shot TimesFM/Chronos.
- **C** — Interactive business dashboard: Taipy/Streamlit or React + Shadcn; CSV ingest, truth-vs-prediction charts, manifest inventory.

Target business use-case: high-value sequential time-series — energy/OZE load
forecasting, industrial IoT predictive maintenance.

## Initial Framing (preserved)

- **User's stated cause or approach**: s-03 "unlocked lightweight session loading", so the next epic should be selected on what that unlock enables; the decision should be judged on engineering impact, architectural coherence, and scope control.
- **User's proposed direction**: evaluate A/B/C, identify the immediate bottleneck, run a tradeoff analysis, pick a winner and propose a minimal 3-phase plan.
- **Pre-dispatch narrowing**: (1) *"No external consumer yet"* — capability/portfolio building; the audience is future clients/employers and the user's own roadmap, nobody is blocked today. (2) *"Order all three"* — all three are likely to happen eventually; the real question is correct sequencing and what each unblocks.

## Dimension Map

The "which epic next" question could originate at any of these dimensions:

1. **The reader-side contract is unexercised** — s-03 built a versioned contract whose only consumers are tests. If this is where the gap lives, the "unlock" is theoretical capability, not delivered capability.
2. **The packaging/dependency boundary is undrawn** — if this is the gap, Option A's *stated benefit* ("lightweight… without dragging training overhead into inference") is not achievable as written, regardless of which epic is chosen. ← partly initial framing
3. **The contract lacks model-kind identity** — if this is the gap, Option B churns the artifact schema, and doing B before A means paying the contract cost twice.
4. **The presentation layer has no data source** — if this is the gap, Option C is not independently buildable and would force ad-hoc scoring logic.
5. **Baseline model quality is the limiting factor** — Option B's implicit premise. If this is the gap, modelling work outranks plumbing.

## Hypothesis Investigation

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| **1. Reader-side contract is unexercised** | `load_session` is defined at `session.py:310` and has **zero production callers** — `grep` across `apps/backend/src/` returns only its own definition; the only callers are `tests/test_session.py` and `tests/test_pipeline.py`. s-03's own frame brief names the motive: the contract "makes a future reader (a predict CLI) possible". `README.md:149` already carries `- [ ] mlpp-predict CLI` as unchecked roadmap item #1. `session.py:316-317` states outright: *"scoring belongs to the predict CLI, not to the contract."* | **STRONG** |
| **2. Packaging boundary undrawn** | `pyproject.toml:9` declares `tensorflow>=2.20,<2.22` as an **unconditional** runtime dependency; there is no `[project.optional-dependencies]` / extras split. `pyproject.toml:19` registers exactly one script, `mlpp-train`. So `load_session()` is import-cheap, but *scoring* requires the Keras model and therefore TF — the "lightweight inference" property holds for contract validation only, not for prediction. | **STRONG** |
| **3. Contract lacks model-kind identity** | Structurally the manifest is already framework-agnostic: `ArtifactEntry` is `{role, filename}` strings (`session.py:120-126`) and `SessionManifest` holds `schema_version`/`created`/`features`/`artifacts` (`session.py:130-144`) — storing a LightGBM booster needs **no** schema change. But no field records *which framework* wrote a model, so a reader must infer the loader from the file extension (`BEST_MODEL_FILE = "best_model.keras"`, `session.py:40`). Tolerable at one framework; blocking at several. | **WEAK** (latent — becomes blocking only under B) |
| **4. Presentation layer has no data source** | There is no scoring path in `src/mlpp/` at all (see hypothesis 1). A dashboard would have to either re-implement scoring or shell out to training. Given the single-owner invariant in `CLAUDE.md`, a second writer/scorer is precisely the failure mode s-03 was built to prevent. | **STRONG** (as a dependency claim) |
| **5. Baseline quality is the limiting factor** | No evidence in-repo. `metrics.py` computes metrics and `test_metrics.csv` is produced per run, but there is **no** comparative benchmark, no baseline-vs-baseline harness, and no recorded observation that Conv1D+BiGRU underperforms on the target use-case. B's premise is currently unmeasured. | **NONE** |

## Narrowing Signals

- **"No external consumer yet."** This removes user-facing urgency as a tiebreaker and makes *demonstrable, exercised capability* the currency. An unused `load_session()` is the weakest possible thing to show an evaluator; a working score path is the strongest.
- **"Order all three."** This converts the question from "which is most valuable" to "which ordering minimises rework". Sequencing turns on what each epic unblocks and what it would force the others to redo.
- **`load_session()` has zero production callers.** Decisive. The capability cited as the reason to choose is itself unexercised.
- **The project already recorded this decision** (`README.md:149`) before the three options were tabled — the frame is partly re-litigating settled ground.
- **`metrics.py` has no comparative harness**, so B cannot currently be justified by a measured deficiency — only by anticipated capability.

## Cross-System Convention

The convention for a contract-plus-reader split is that the contract is not
considered done until one real consumer exercises it end to end; tests
exercising a loader is a unit-level guarantee, not an integration one. This
project already follows that convention explicitly — s-03's frame justified the
manifest by the reader it would enable, and `session.py:316-317` reserves
scoring for a predict CLI by name. The leading hypothesis matches the
convention: the missing consumer is the next unit of work.

Notably, the remaining gap is *small*. Three of the four steps a predict path
needs already exist and are TF-free: `load_session()` restores manifest +
fitted `Preprocessor` (`session.py:310`), `Preprocessor.transform` already
returns `y=None` for target-less inference input (`preprocess.py:164-165`), and
`Preprocessor.inverse_target` already un-scales predictions back to engineering
units (`preprocess.py:195-199`). The only missing link is loading the Keras
model and calling `predict`.

## Reframed (or Confirmed) Problem Statement

> **The actual problem to plan around is**: s-03 delivered a versioned artifact
> contract that nothing in production reads — the engine can train and persist a
> session but cannot score a new CSV — so the next epic must close the
> reader-side seam, and the choice between A/B/C is really a sequencing question
> whose answer is forced by that gap.

The initial framing was **directionally right but mis-stated in two ways**, and
both matter for planning:

1. **The "unlock" is not yet an unlock.** `load_session()` has no production caller. s-03 created the *possibility* of a reader; it did not create a reader. Choosing B or C next would leave that contract unexercised while adding load on top of it — B widens what a session must describe, C consumes a scoring path that does not exist.
2. **"Lightweight inference" is not currently true, and no epic gets it for free.** TF is an unconditional runtime dependency (`pyproject.toml:9`). Scoring needs the Keras model, so an inference deployment installs the full training stack today. A's *value* is real; A's *stated benefit* needs an explicit packaging decision to become real. That decision belongs in the plan, not assumed away.

What changes if this is addressed: the manifest contract gets its first real
consumer and stops being unverified structure; the README's roadmap item #1
closes; B inherits a settled reader to extend rather than a hypothetical one;
and C gets an actual data source instead of a reason to grow a second scorer.

## Confidence

**HIGH** — the decisive evidence is a zero-caller `grep` on `load_session`,
corroborated independently by three sources that predate the question
(`session.py:316-317`, `README.md:149`, s-03's own frame brief), with the
competing dimension (hypothesis 5, "baseline quality is the limit") ruled out by
the absence of any comparative benchmark in the repo. The pressure-test
strengthened rather than weakened the hypothesis, and additionally surfaced a
defect in the *premise* of the winning option (the TF packaging boundary) that
had not been checked.

## What Changes for /10x-plan

Plan **Option A, narrowly**: the scoring path, not the whole platform. Sequence
**A → B → C**. The plan's subject is "give the manifest contract its first
production reader", not "build an API".

Non-negotiable invariants the plan must hold (all from `CLAUDE.md` + s-03):

- **One owner for session artifacts.** Any file the predict path writes goes through `SessionWriter.register(role, filename)`; no module outside `session.py` spells a filename. A predict path that writes ad-hoc output files re-creates the exact drift s-03 removed.
- **`manifest.json` stays the single source of truth** for the column contract. The predict path reads the feature contract from the manifest — never from `input_columns` order, never from a second persisted copy.
- **Feature order is not config order.** Locate columns with `preprocessor.index_of(name)`.
- **Keep the TF-free layers TF-free.** `config`, `data`, `preprocess`, `metrics`, `artifacts` and `session` must not gain a TensorFlow import. Model loading belongs in a new TF-owning module, so the fast suite stays fast.
- **Artifacts are rejected, not migrated.** Version mismatch raises `SchemaVersionError`; no compatibility shims.
- **Errors from the `MlppError` hierarchy**; never `None` to signal failure (the `transform` y=None inference case is the sole existing exception).

Two decisions the plan must make explicitly rather than inherit:

- **Packaging boundary** — either accept TF in the inference install (simple, honest, ~600 MB) or introduce extras to separate scoring from training. This determines whether "lightweight FastAPI" is truthful.
- **Model-kind in the manifest** — leave the loader inferred from extension now (fine at one framework) and accept a `SCHEMA_VERSION` bump when B lands, or add the field now. Deferring is defensible; doing it blind is not.

## References

- Source files: `apps/backend/src/mlpp/session.py:310` (`load_session`, zero production callers), `session.py:316-317` (predict-CLI reservation), `session.py:120-144` (framework-agnostic manifest), `session.py:40` (`BEST_MODEL_FILE`), `apps/backend/src/mlpp/preprocess.py:164-165` (`transform` y=None), `preprocess.py:195-199` (`inverse_target`), `apps/backend/src/mlpp/model.py:1-49` (Keras-only builder), `apps/backend/pyproject.toml:9` (unconditional TF dep), `pyproject.toml:19` (only `mlpp-train` registered), `README.md:149` (roadmap item #1)
- Related prior decision: `context/archive/2026-07-29-s-03-algorithm-and-architecture-deep-optimization/frame.md` (the "future reader (a predict CLI)" rationale)
- Investigation: conducted by direct read of the 12 modules in `src/mlpp/` (surface small and recently worked; no sub-agents dispatched)

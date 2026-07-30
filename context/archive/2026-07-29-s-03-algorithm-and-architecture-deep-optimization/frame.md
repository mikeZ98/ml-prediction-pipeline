# Frame Brief: mlpp engine "deep optimization"

> Framing step before /10x-plan. This document captures what is *actually*
> at issue, separated from what was initially assumed.

## Reported Observation

**No observable was reported.** The request (`change.md:12`) names three techniques and
one quality goal — "refactor mlpp engine for performance, memory-mapped I/O,
vectorization, and clean domain-driven abstractions" — but no measured effect: no
duration, no memory ceiling, no failing operation, no scale threshold.

The empty observation slot is the primary finding of this frame. There was no fixed
ground to pin the dimension map to until Step 1.5 supplied it.

## Initial Framing (preserved)

- **User's stated cause or approach**: the mlpp engine needs performance work; I/O and
  vectorization are the levers; the abstractions aren't clean.
- **User's proposed direction**: memory-mapped I/O, vectorization, domain-driven abstractions.
- **Pre-dispatch narrowing**: observable = *"Nothing measured — code-shape intuition."*
  Scope = *"One — architecture is the goal"* (performance is a hoped-for side effect).

That answer alone demotes the two performance techniques from *requirements* to
*guesses*, and promotes architecture from a co-equal item to **the** goal.

## Dimension Map

The dissatisfaction could originate at any of these dimensions:

1. **Artifact & persistence ownership** — fitted state has no single owner or contract.
2. **`Preprocessor` responsibility count** — one class spans schema, scaling, encoding,
   name bookkeeping, and persistence.
3. **Data-path performance** — I/O and vectorization are real bottlenecks. ← initial framing
4. **Framework coupling** — domain logic entangled with sklearn/Keras.
5. **Orchestration shape** — `pipeline.py` mixes discovery, training, evaluation, persistence.

## Hypothesis Investigation

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| **1. Persistence has no owner or contract** | `feature_names` is written by **three** modules to **three** files: `preprocess.py:211`→`encoders.gz`, `artifacts.py:40`→`feature_config.json`, `pipeline.py:121`→`model_iter_NN_config.json`. No cross-file consistency check exists anywhere in `src/mlpp`. No schema version field. In production nothing *reads* any of them — `load_feature_config` and `Preprocessor.load` are called only from tests (`test_artifacts.py:41`, `test_preprocess.py:139`). **The committed reference run `OUTPUTS/example/` is already rejected by the current loader**: `ArtifactError: missing keys: ['feature_names', 'numeric_columns']` — it is still in the pre-refactor notebook schema, and nothing detects that but a runtime failure. | **STRONG** |
| **2. `Preprocessor` does too much** | 13 public members across 5 concerns (`preprocess.py:96-232`): schema resolution, X scaling, y scaling, one-hot encoding, feature-name bookkeeping, and joblib persistence. Largest module in the package at 235 lines. Persistence is the concern that clearly doesn't belong — and it is the same concern as hypothesis 1, seen from the other side. | **STRONG** (largely derivative of 1) |
| **3. Data-path performance (initial framing)** | Measured end-to-end on the real sample data: CSV read **0.0139s**, `fit` **0.0610s**, `transform` **0.0010s** — total data path **0.076s = 0.5% of runtime**. TensorFlow import alone is **11.02s**, 145× the entire data path. Feature matrix `X` is **76.2 KB**. Preprocessing is already fully vectorized (sklearn ops over whole frames; no row loops). Total input on disk: 760K train + 408K test. | **NONE** |
| **4. Framework coupling** | Layering is already clean: `errors.py` and `config.py` have zero internal deps; TensorFlow is confined to `model/training/plots/pipeline`; `config·data·preprocess·metrics·artifacts` import no TF at all (which is what makes the 59-test fast suite run in ~1s). No circular imports. | **WEAK** |
| **5. Orchestration shape** | `pipeline.py` is 182 lines / 6 defs and does mix discovery, fitting, training, evaluation and artifact writing. Real, but it is a composition root — mixing is partly its job, and the messiness it exhibits is mostly *downstream* of hypothesis 1. | **WEAK–MEDIUM** |

## Narrowing Signals

- **User: "nothing measured — code-shape intuition."** Decisive. Rules out dimension 3 as a
  *reported* problem; whatever remains must be justified on design grounds, not speed.
- **User: "architecture is the goal."** Collapses the bundled request to a single concern
  and makes "performance" an outcome to be tested, not a requirement to be met.
- **Measurement: data path = 0.5% of runtime, `X` = 76 KB.** Independently confirms
  dimension 3 is not merely unreported but *not present*. Memory-mapping a 76 KB array
  and vectorizing a 1 ms call cannot produce a win.
- **The committed `OUTPUTS/example/` fails today's loader.** Not predicted before
  pressure-testing; found by checking what hypothesis 1 implies but hadn't been verified.

## Cross-System Convention

Fitted-state persistence is conventionally owned by **one** component with a **versioned**
schema — sklearn pipelines pickle a single estimator; MLflow writes one `MLmodel` manifest
with a format version. This codebase instead has three writers, no version field, and no
reader in production. The leading hypothesis matches the convention's failure mode exactly:
duplicated fields drift, and nothing notices until a load fails.

## Reframed (or Confirmed) Problem Statement

> **The actual problem to plan around is**: the engine has no single owner and no versioned
> contract for fitted-state artifacts — three modules write overlapping copies of the same
> facts, nothing reads them in production, and the one committed reference run is already
> incompatible with the current loader.

This is what the "clean abstractions" intuition is correctly pointing at. It is also why the
`Preprocessor` class feels overloaded: persistence is the responsibility that doesn't belong
there, and it is the same defect viewed from the class rather than the file. Addressing it
gives the artifact directory a schema that can be validated and versioned, makes a future
reader (a predict CLI) possible without arbitrarily picking one of three sources of truth,
and shrinks `Preprocessor` as a side effect.

The performance half of the initial framing does **not** survive: it was not driven by an
observation, and measurement shows the data path is 0.5% of runtime on a 76 KB matrix.

## Confidence

**HIGH** — strong evidence with file:line references, a decisive user narrowing signal, a
measurement that conclusively rules out the competing dimension, and a pressure-test that
surfaced a defect the hypothesis predicted but had not been checked.

## What Changes for /10x-plan

Plan an **artifact-contract change**, not a performance change: one owner for fitted-state
persistence, a versioned and validated schema, and a decision on what to do with the now-stale
`OUTPUTS/example/`. Do **not** plan memory-mapped I/O or vectorization — the measurement
rules them out. Any `Preprocessor` decomposition should be justified as a consequence of
moving persistence out, not as an independent goal.

## References

- Request: `context/changes/s-03-algorithm-and-architecture-deep-optimization/change.md:12`
- Three writers: `src/mlpp/preprocess.py:211`, `src/mlpp/artifacts.py:40`, `src/mlpp/pipeline.py:121`
- Test-only readers: `tests/test_artifacts.py:41`, `tests/test_preprocess.py:139`
- Overloaded class: `src/mlpp/preprocess.py:96-232`
- Stale committed run: `OUTPUTS/example/feature_config.json` (pre-refactor schema)
- Investigation: performed inline (1096-line codebase, authored this session — sub-agents
  not dispatched per the skill's "use when the surface is large or unfamiliar" guidance)

# Session Artifact Contract — Plan Brief

> Full plan: `context/changes/s-03-algorithm-and-architecture-deep-optimization/plan.md`
> Frame brief: `context/changes/s-03-algorithm-and-architecture-deep-optimization/frame.md`

## What & Why

The engine has no single owner and no versioned contract for fitted-state artifacts — three
modules write overlapping copies of the same facts, nothing reads them in production, and the
one committed reference run is already incompatible with the current loader. This change gives
the session directory one owner and a versioned manifest, then proves it by reading it back.

## Starting Point

`feature_names` is persisted three times (`preprocess.py:211`, `artifacts.py:40`,
`pipeline.py:121`) with no consistency check; artifact filenames are spread across four
modules plus three inline f-strings; and `load_feature_config` / `Preprocessor.load` have
only test callers. The committed `OUTPUTS/example/` already fails today's loader with
`missing keys: ['feature_names', 'numeric_columns']` — proof the drift is real, not theoretical.

## Desired End State

A session directory is self-describing: one `manifest.json` carries `schema_version`, the
feature contract, and an inventory of every file with its role, written by the only module
allowed to name files in that directory. `load_session()` restores a working preprocessor from
it, and artifacts from older code fail with an error naming both versions instead of a
`KeyError`.

## Key Decisions Made

| Decision | Choice | Why (1 sentence) | Source |
| --- | --- | --- | --- |
| Is this a performance change? | No — architecture only | Measured: data path is 0.5% of runtime on a 76 KB matrix; TF import alone is 145× larger. | Frame |
| Real defect | Artifact ownership, not code shape | Three writers, zero production readers, no version field, committed example already broken. | Frame |
| Contract shape | Single `manifest.json`, one owning module | Kills triplication at the root and makes the directory self-describing; MLflow's `MLmodel` proves the pattern. | Plan |
| Legacy artifacts | Fail loudly with a version error | Artifacts regenerate in ~10s, so a permanent shim would be ongoing cost for a one-time problem. | Plan |
| Stale `OUTPUTS/example/` | Regenerate under the new contract | Ships an example a reader can actually load, keeping README's table honest. | Plan |
| Reader scope | Minimal load-and-verify, no CLI | An unexercised contract is how the example drifted; scoring belongs to the predict change. | Plan |
| `Preprocessor` decomposition | Move persistence out, stop there | The only concern the frame's evidence implicates; further splitting is speculative. | Plan |

## Scope

**In scope:** new `session.py` owning all session filenames and the manifest; version-aware
load with `SchemaVersionError`; removal of `Preprocessor.save`/`load`, `feature_config.json`
and `model_iter_NN_config.json`; migration of four coupled test modules; round-trip tests;
regenerated `OUTPUTS/example/`; README and CLAUDE.md updates.

**Out of scope:** memory-mapped I/O and vectorization (measured out); `mlpp-predict` CLI;
further `Preprocessor` splitting; legacy-format reading or migration scripts; model, training,
metrics and plot changes.

## Architecture / Approach

Build the owner in isolation first, then invert the dependencies: modules stop naming their own
files and register with the owner instead. Reading only becomes meaningful once writes are
consolidated, so the load path follows. Docs and the regenerated example come last, describing
what actually shipped. The manifest is JSON rather than pickle so it stays diffable and
readable without importing sklearn — consistent with keeping `session.py` TensorFlow-free.

## Phases at a Glance

| Phase | What it delivers | Key risk |
| --- | --- | --- |
| 1. Session manifest module | `session.py` + manifest schema, versioned read/write, tests | Getting the schema shape wrong — later phases depend on it |
| 2. Route writers through owner | Single writer; triplication and side-cars gone | Only phase touching existing behaviour; 4 source + 4 test modules move at once |
| 3. Load-and-verify path | `load_session()` + round-trip proof | An API with only test callers until the predict change lands |
| 4. Regenerate run + docs | Valid `OUTPUTS/example/`, accurate README and CLAUDE.md | Rewrites ~24MB of committed binaries; numerics differ from the original |

**Prerequisites:** none beyond `main` at `6e7d349` — the package, its 77 tests and CI are in place.
**Estimated effort:** ~2–3 sessions across 4 phases; phase 2 is the bulk.

## Open Risks & Assumptions

- Phase 2 changes four source and four test modules together. The plan asserts net test count
  must not fall, but coverage could thin silently if cases are dropped rather than moved.
- The manifest schema is fixed in phase 1 and depended on by phases 2–4; getting a field wrong
  is the one expensive-rework path here.
- Regenerating the example rewrites committed binaries and produces different numerics than the
  README's original figures imply. Nothing asserts those figures, but a reader may notice.
- Assumes no external consumer reads `feature_config.json` outside this repo. Nothing in the
  repo does, but that cannot be verified beyond it.

## Success Criteria (Summary)

- A fresh `mlpp-train` run produces a directory whose `manifest.json` states the feature
  contract exactly once, and `load_session()` reconstructs a working preprocessor from it.
- Pointing the loader at a pre-change session directory yields an error naming the version
  found and the version expected — not a traceback.
- `OUTPUTS/example/` loads cleanly, and README's artifact table matches its contents file for file.

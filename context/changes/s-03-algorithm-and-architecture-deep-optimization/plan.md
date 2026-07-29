# Session Artifact Contract Implementation Plan

## Overview

Fitted-state artifacts currently have no owner and no contract. Three modules write
overlapping copies of the same facts into a session directory, nothing reads them in
production, and the committed reference run is already incompatible with the current loader.

This plan introduces a single **session manifest** with a schema version, routes every
session-directory write through one owning module, and adds a minimal load path that proves
the contract holds.

## Current State Analysis

Per `frame.md`, confirmed by direct inspection:

- **Filenames are scattered across four modules**: `artifacts.py:12`, `pipeline.py:25`,
  `preprocess.py:23-25`, `training.py:14-15`. Three more filenames exist only as inline
  f-strings in `pipeline.py:116-119` (`history_train_NN.csv`, `model_iter_NN.keras`,
  `model_iter_NN_config.json`).
- **`feature_names` is written three times**: `preprocess.py:211`→`encoders.gz`,
  `artifacts.py:40`→`feature_config.json`, `pipeline.py:121`→`model_iter_NN_config.json`.
  No code anywhere cross-checks them.
- **No production reader exists.** `load_feature_config` and `Preprocessor.load` have
  test-only callers (`test_artifacts.py:41`, `test_preprocess.py:139`).
- **The committed `OUTPUTS/example/` is already broken** against today's loader:
  `ArtifactError: missing keys: ['feature_names', 'numeric_columns']`. It is in the
  pre-refactor notebook schema and nothing detects that but a runtime failure.
- **`Preprocessor` spans five concerns** across 13 public members (`preprocess.py:96-232`);
  persistence is the one that does not belong.
- The data path is **0.5% of runtime** (0.076s; `X` is 76.2 KB). Performance is not in play.

Constraints to work within: `config`/`data`/`preprocess`/`metrics`/`artifacts` must stay
TensorFlow-free (that is what makes the 59-test fast suite run in ~1s); errors raise from the
`MlppError` hierarchy; `mypy --strict` and `ruff` must stay clean.

## Desired End State

A session directory is self-describing. One module owns every filename in it and writes a
single `manifest.json` carrying `schema_version`, the feature contract, and an inventory of
the files present with their roles. A `load_session()` reads that manifest, validates the
version, and reconstructs a working `Preprocessor` — proven by a round-trip test. Artifacts
written by older code fail with a specific, actionable error rather than a `KeyError`.

Verify by: `uv run pytest` green; `uv run mlpp-train` produces a directory whose manifest
`load_session()` accepts; loading `OUTPUTS/example/` from before this change raises a
version error naming both versions.

### Key Discoveries:

- Three writers, zero production readers — `preprocess.py:211`, `artifacts.py:40`,
  `pipeline.py:121`.
- The stale example proves the failure mode is real, not theoretical
  (`OUTPUTS/example/feature_config.json` holds `input_columns`, not `feature_names`).
- `test_pipeline.py:26` asserts literal artifact filenames; `test_artifacts.py` is built
  entirely around `FEATURE_CONFIG_FILE`. Both move with the contract.
- `artifacts.py` already imports `FeatureSchema` from `preprocess.py` — the persistence
  concern is *already* split across two modules, inconsistently.

## What We're NOT Doing

- **No memory-mapped I/O and no vectorization work.** Ruled out by measurement in `frame.md`.
- **No `mlpp-predict` CLI.** The load path stops at "restore and validate"; scoring, output
  formats and batching belong to a separate change.
- **No further `Preprocessor` decomposition.** Scaling and encoding stay where they are;
  only persistence moves out.
- **No backwards-compatible reading of legacy artifacts.** No shim, no migration script.
- **No changes to model architecture, training loop, metrics, or plots.**
- **No `.keras`/`.png`/`.html` format changes** — only who names and records them.

## Implementation Approach

Build the owner first in isolation (phase 1), so it is fully tested before anything depends
on it. Then invert the dependencies: writers stop naming their own files and ask the owner
instead (phase 2). Only once writes are consolidated does reading become meaningful (phase 3).
Documentation and the regenerated reference run come last, because they must describe what
actually shipped (phase 4).

The manifest is JSON, not pickle, so it stays diffable, inspectable, and readable without
importing sklearn — consistent with the existing choice to keep `artifacts.py` TF-free.

## Critical Implementation Details

**Timing & lifecycle.** The manifest can only be written *after* the preprocessor is fitted
(feature names do not exist before `fit`) but the session directory is created *before*
training starts. So the owner must create the directory eagerly and write the manifest at
first-fit — not at construction. `pipeline.py:73-75` is where that ordering currently lives.

**State sequencing.** `run_pipeline` writes stage artifacts inside the per-stage loop
(`pipeline.py:116-119`) and metrics after the loop (`pipeline.py:102`). The manifest's file
inventory must therefore be appended to across stages and flushed at the end, or it will
record only the first stage. The obvious "write it once when fitting" ordering is wrong.

## Phase 1: Session manifest module

### Overview

Introduce the module that owns the session-directory contract. Nothing calls it yet; this
phase is pure construction plus tests.

### Changes Required:

#### 1. Manifest schema and owner

**File**: `apps/backend/src/mlpp/session.py` (new)

**Intent**: Own every session-directory filename and the manifest that describes them, so no
other module ever names a file in that directory again. Replaces the ownership vacuum that
let `feature_names` triplicate.

**Contract**: A `SessionManifest` dataclass carrying `schema_version: int`, the feature
contract (numeric columns, categorical columns, output column, expanded feature names), and
an artifact inventory mapping role → filename. A `SessionWriter` owning `session_dir`, with
methods to register artifacts by role and flush the manifest. Filename constants for every
artifact currently defined in `artifacts.py:12`, `pipeline.py:25`, `preprocess.py:23-25`,
`training.py:14-15`, plus the three inline patterns from `pipeline.py:116-119`. Manifest file
is `manifest.json`. `SCHEMA_VERSION` starts at 1.

Must not import TensorFlow — this module belongs to the fast test suite.

#### 2. Version-aware load and validation

**File**: `apps/backend/src/mlpp/session.py`

**Intent**: Read a manifest back and reject anything this code cannot honestly interpret,
with an error that tells the operator what to do rather than surfacing a `KeyError`.

**Contract**: A `read_manifest(session_dir) -> SessionManifest` that raises `ArtifactError`
when the file is absent, is not valid JSON, is not an object, lacks `schema_version`, or
carries a version this build does not support. The unsupported-version message must name both
the version found and the version expected. A directory in the pre-manifest layout (no
`manifest.json`) must produce the same clear failure, not a missing-file traceback.

#### 3. Error type for version mismatch

**File**: `apps/backend/src/mlpp/errors.py`

**Intent**: Give version rejection its own type so callers can distinguish "this is an old
artifact" from "this artifact is corrupt".

**Contract**: A `SchemaVersionError` subclassing `ArtifactError`, exported from
`mlpp/__init__.py` alongside the existing hierarchy.

#### 4. Unit tests

**File**: `apps/backend/tests/test_session.py` (new)

**Intent**: Cover the contract before anything depends on it.

**Contract**: Round-trip of manifest write→read; rejection of absent, malformed, non-object,
version-less, and future-version manifests; assertion that a directory with no `manifest.json`
raises `SchemaVersionError` or `ArtifactError` with an actionable message; artifact-inventory
registration across multiple stages.

### Success Criteria:

#### Automated Verification:

- Fast suite passes: `cd apps/backend && uv run pytest -m 'not slow'`
- New module imports no TensorFlow: `uv run python -c "import sys, mlpp.session; assert 'tensorflow' not in sys.modules"`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`

#### Manual Verification:

- The manifest JSON is human-readable and its field names are self-explanatory when opened in an editor

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to the next phase.

---

## Phase 2: Route all writers through the owner

### Overview

Invert the dependencies. Every module that currently names a file in the session directory
stops doing so and registers with the owner instead. The triplication disappears because
there is only one place left that can write feature names.

### Changes Required:

#### 1. Remove persistence from Preprocessor

**File**: `apps/backend/src/mlpp/preprocess.py`

**Intent**: Move `save`/`load` and the three filename constants out to the session owner,
leaving the class responsible only for turning a dataframe into `X` and `y`. This is the
decomposition the frame's evidence supports — no other concern moves.

**Contract**: `Preprocessor.save` and `Preprocessor.load` are removed, along with
`SCALER_FILE`, `OUTPUT_SCALER_FILE`, `ENCODERS_FILE` (`preprocess.py:23-25`). The class
exposes its fitted internals to the session owner through a narrow accessor rather than
writing files itself. `fit`, `transform`, `fit_transform`, `inverse_target`, `index_of`,
`feature_names`, `schema`, `n_features` and `is_fitted` all keep their current signatures —
callers of those are unaffected.

#### 2. Collapse artifacts.py into the session owner

**File**: `apps/backend/src/mlpp/artifacts.py`

**Intent**: `feature_config.json` and its loader are superseded by the manifest. Keep
`make_session_dir` (still needed) and delete the duplicated feature-contract persistence.

**Contract**: `save_feature_config`, `load_feature_config` and `FEATURE_CONFIG_FILE` are
removed; `feature_config.json` is no longer written. `make_session_dir` either stays here or
moves into `session.py` — implementer's call, but it must have exactly one home. The
`FeatureSchema` import from `preprocess.py` goes away with the functions that used it.

#### 3. Pipeline stops naming files

**File**: `apps/backend/src/mlpp/pipeline.py`

**Intent**: Remove the inline f-string filenames and the `model_iter_NN_config.json`
side-car, whose entire content is now in the manifest.

**Contract**: `METRICS_FILE` (`pipeline.py:25`) moves to the session owner. The three inline
patterns at `pipeline.py:116-119` are replaced by registration calls. `model_iter_NN_config.json`
is no longer written — `n_features` and `feature_names` live in the manifest only.
`run_pipeline`'s signature and `PipelineResult` are unchanged. Per Critical Implementation
Details, the inventory accumulates across stages and is flushed once at the end.

#### 4. Training filename constants

**File**: `apps/backend/src/mlpp/training.py`

**Intent**: `BEST_MODEL_FILE` and `TRAINING_LOG_FILE` (`training.py:14-15`) are session-directory
names and belong to the owner, even though this module imports Keras.

**Contract**: Both constants move to `session.py`; `training.py` imports them. This keeps
`session.py` TF-free (constants are strings) while removing the last filenames defined outside
the owner.

#### 5. Migrate coupled tests

**File**: `apps/backend/tests/test_artifacts.py`, `test_preprocess.py`, `test_pipeline.py`, `test_model.py`

**Intent**: Four test modules assert filenames or call the removed persistence API. They move
to the new contract without losing coverage.

**Contract**: `test_artifacts.py` loses its `FEATURE_CONFIG_FILE` cases (superseded by
`test_session.py`) and keeps only `make_session_dir` coverage — or is deleted if that moves.
`test_preprocess.py:139,151` (save/load round-trip) moves to `test_session.py`.
`test_pipeline.py:26` asserts manifest-era filenames. `test_model.py:103-112` imports the
constants from their new home. Net test count must not fall.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- No filename literals remain outside the owner: `! grep -rnE '"[a-z_]+\.(gz|json|keras)"' src/mlpp --include='*.py' | grep -v session.py`
- Type checking passes: `uv run mypy src/mlpp`
- Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .`
- A real run still completes: `uv run mlpp-train --epochs 1 --no-plots --quiet`

#### Manual Verification:

- A freshly produced session directory contains `manifest.json` and no `feature_config.json` or `model_iter_NN_config.json`
- Opening the manifest shows the feature contract exactly once, with no duplicated fields

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to the next phase.

---

## Phase 3: Load-and-verify path

### Overview

Give the contract a reader. Without one it cannot be falsified — which is precisely how the
committed example drifted out of compatibility unnoticed.

### Changes Required:

#### 1. Session load path

**File**: `apps/backend/src/mlpp/session.py`

**Intent**: Reconstruct a usable `Preprocessor` from a session directory, validating that the
manifest and the files on disk agree.

**Contract**: A `load_session(session_dir, cfg) -> LoadedSession` returning the manifest and a
fitted `Preprocessor` ready for `transform`. Raises `ArtifactError` when a file the manifest
lists is missing from disk, and `SchemaVersionError` on version mismatch. Does not load the
Keras model — that is the predict change's concern and would drag TensorFlow into this module.

#### 2. Round-trip test

**File**: `apps/backend/tests/test_session.py`

**Intent**: Prove write→read agreement end to end, which is the assertion that would have
caught the stale example.

**Contract**: Fit a preprocessor, write a session, load it back, and assert
`transform()` on the same frame produces numerically identical `X` and `y` to the original —
plus `feature_names` equality. Also assert that deleting a manifest-listed file causes
`load_session` to raise rather than silently degrade.

#### 3. Pipeline-level round trip

**File**: `apps/backend/tests/test_pipeline.py`

**Intent**: Verify the contract against a directory produced by the real pipeline, not a
hand-built fixture.

**Contract**: A `slow`-marked test running `run_pipeline`, then `load_session` on
`result.session_dir`, asserting the restored preprocessor reproduces the pipeline's feature
names and `n_features`.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- Round-trip test present and passing: `uv run pytest -k round_trip -v`
- `session.py` still imports no TensorFlow: `uv run python -c "import sys, mlpp.session; assert 'tensorflow' not in sys.modules"`
- Type checking passes: `uv run mypy src/mlpp`

#### Manual Verification:

- Pointing `load_session` at the old pre-change `OUTPUTS/example/` produces an error naming both the version found and the version expected — not a traceback

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful before proceeding to the next phase.

---

## Phase 4: Regenerate reference run and update docs

### Overview

Replace the stale committed example with one written under the new contract, and update the
documents that describe the artifact layout.

### Changes Required:

#### 1. Regenerate the reference run

**File**: `OUTPUTS/example/`

**Intent**: Ship an example a reader can actually load, so the contract is demonstrated
rather than merely described.

**Contract**: Delete the existing directory and regenerate via `uv run mlpp-train` with plots
enabled. `load_session` must accept the result. Note the diff rewrites ~24MB of committed
binaries and numerics will differ from the original notebook run.

#### 2. README artifact table

**File**: `README.md`

**Intent**: The documented output list is now wrong in three places.

**Contract**: Line 7 (`feature_config.json` in the feature bullet) and the table at lines
94-96 update to describe `manifest.json` and drop `feature_config.json` and
`model_iter_NN_config.json`. Add a short note that the manifest carries `schema_version` and
that artifacts from older versions are rejected rather than coerced.

#### 3. CLAUDE.md domain invariant

**File**: `CLAUDE.md`

**Intent**: Record the ownership rule so a future agent does not reintroduce a second writer —
the exact regression this change exists to prevent.

**Contract**: Add to the `## Domain invariants` section (line 54): every session-directory
filename is owned by `session.py`; no other module names files in a session directory; the
manifest is the single source of truth for the feature contract.

### Success Criteria:

#### Automated Verification:

- Full suite passes: `cd apps/backend && uv run pytest`
- Regenerated example loads: `uv run python -c "from mlpp.session import load_session; from mlpp.config import PipelineConfig; from pathlib import Path; r=Path('../..').resolve(); load_session(r/'OUTPUTS/example', PipelineConfig.for_repo(r).data); print('OK')"`
- Example contains a manifest: `test -f OUTPUTS/example/manifest.json`
- No stale side-cars remain: `! ls OUTPUTS/example/feature_config.json OUTPUTS/example/model_iter_01_config.json 2>/dev/null`
- CI passes on the branch (both Python versions plus notebook-drift)

#### Manual Verification:

- README's artifact table matches the actual contents of the regenerated `OUTPUTS/example/`, file for file
- CLAUDE.md's new invariant reads clearly enough that an agent would not add a second writer

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the manual testing was successful.

---

## Testing Strategy

### Unit Tests:

- Manifest round-trip: write then read yields an equal manifest
- Rejection cases: absent file, malformed JSON, non-object payload, missing `schema_version`, unsupported version
- Pre-manifest directory produces an actionable error, not a traceback
- Artifact inventory accumulates correctly across multiple training stages
- Preprocessor restored from a session transforms identically to the original

### Integration Tests:

- `run_pipeline` → `load_session` on the produced directory, asserting feature-name and `n_features` agreement (marked `slow`)
- A full `mlpp-train` run still produces every artifact the README lists

### Manual Testing Steps:

1. Run `uv run mlpp-train --epochs 1` and open the resulting `manifest.json` — confirm the feature contract appears exactly once and the file inventory lists every file present.
2. Point `load_session` at a session directory saved before this change; confirm the error names both versions and suggests regenerating.
3. Delete one file listed in a valid manifest, re-run `load_session`, and confirm it raises rather than degrading silently.

## Performance Considerations

None. The frame measured the data path at 0.076s — 0.5% of runtime, on a 76.2 KB feature
matrix — against 11.02s for the TensorFlow import alone. Adding one small JSON write and read
per session is not measurable. **No performance work is in scope**, and any proposal to add
some should be re-measured first.

## Migration Notes

Deliberately no migration path. Artifacts regenerate in roughly ten seconds, so permanent
compatibility code would be ongoing cost for a one-time problem. Existing session directories —
including the committed example, which is *already* unreadable — stop loading and must be
regenerated. `SCHEMA_VERSION` starts at 1; future breaking changes bump it and the loader
rejects anything it does not recognise.

## References

- Frame brief: `context/changes/s-03-algorithm-and-architecture-deep-optimization/frame.md`
- Three writers: `src/mlpp/preprocess.py:211`, `src/mlpp/artifacts.py:40`, `src/mlpp/pipeline.py:121`
- Test-only readers: `tests/test_artifacts.py:41`, `tests/test_preprocess.py:139`
- Filename constants: `artifacts.py:12`, `pipeline.py:25`, `preprocess.py:23-25`, `training.py:14-15`
- Inline filenames: `pipeline.py:116-119`
- Docs to update: `README.md:7,94-96`, `CLAUDE.md:54`

## Progress

> Convention: `- [ ]` pending, `- [x]` done. Append ` — <commit sha>` when a step lands. Do not rename step titles. See `references/progress-format.md`.

### Phase 1: Session manifest module

#### Automated

- [x] 1.1 Fast suite passes: `cd apps/backend && uv run pytest -m 'not slow'` — 845bb60
- [x] 1.2 New module imports no TensorFlow — 845bb60
- [x] 1.3 Type checking passes: `uv run mypy src/mlpp` — 845bb60
- [x] 1.4 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — 845bb60

#### Manual

- [x] 1.5 The manifest JSON is human-readable and its field names are self-explanatory when opened in an editor — 845bb60

### Phase 2: Route all writers through the owner

#### Automated

- [x] 2.1 Full suite passes: `cd apps/backend && uv run pytest` — e0e6825
- [x] 2.2 No filename literals remain outside the owner — e0e6825
- [x] 2.3 Type checking passes: `uv run mypy src/mlpp` — e0e6825
- [x] 2.4 Linting and formatting pass: `uv run ruff check . && uv run ruff format --check .` — e0e6825
- [x] 2.5 A real run still completes: `uv run mlpp-train --epochs 1 --no-plots --quiet` — e0e6825

#### Manual

- [x] 2.6 A freshly produced session directory contains `manifest.json` and no `feature_config.json` or `model_iter_NN_config.json` — e0e6825
- [x] 2.7 Opening the manifest shows the feature contract exactly once, with no duplicated fields — e0e6825

### Phase 3: Load-and-verify path

#### Automated

- [x] 3.1 Full suite passes: `cd apps/backend && uv run pytest` — 61b1fe6
- [x] 3.2 Round-trip test present and passing: `uv run pytest -k round_trip -v` — 61b1fe6
- [x] 3.3 `session.py` still imports no TensorFlow — 61b1fe6
- [x] 3.4 Type checking passes: `uv run mypy src/mlpp` — 61b1fe6

#### Manual

- [x] 3.5 Pointing `load_session` at the old pre-change `OUTPUTS/example/` produces an error naming both the version found and the version expected — 61b1fe6

### Phase 4: Regenerate reference run and update docs

#### Automated

- [x] 4.1 Full suite passes: `cd apps/backend && uv run pytest`
- [x] 4.2 Regenerated example loads via `load_session`
- [x] 4.3 Example contains a manifest: `test -f OUTPUTS/example/manifest.json`
- [x] 4.4 No stale side-cars remain in `OUTPUTS/example/`
- [ ] 4.5 CI passes on the branch (both Python versions plus notebook-drift)

#### Manual

- [x] 4.6 README's artifact table matches the actual contents of the regenerated `OUTPUTS/example/`, file for file
- [x] 4.7 CLAUDE.md's new invariant reads clearly enough that an agent would not add a second writer

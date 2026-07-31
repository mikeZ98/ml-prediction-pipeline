---
project: mlpp
assessed_at: 2026-07-31T09:41:46Z
agent_readiness: ready-with-compensation
context_type: brownfield
stack_components:
  language: Python 3.12–3.13
  framework: Keras 3 / TensorFlow 2.20–2.21 (ML); Streamlit 1.x (UI, incoming)
  build_tool: hatchling
  test_runner: pytest 8.x
  package_manager: uv
  ci_provider: GitHub Actions
  deployment_target: null
gates_passed: 11
gates_failed: 1
gates_partial: 2
---

# Stack Assessment — mlpp

Assessed against the four agent-friendly criteria, in the context of the `mlpp Dashboard` change
described in `context/foundation/prd.md`.

Two scoping notes. The project marker sits at `apps/backend/pyproject.toml`, not the repository
root — this matches the layout `CLAUDE.md` documents and is not a defect. And Streamlit is scored as
an **incoming** component: the dependency is declared in `pyproject.toml` but the change is
uncommitted and no interface code exists yet, so its scores describe risk ahead rather than
friction today.

## Stack Components

**Language — Python 3.12–3.13.** Constrained by `requires-python = ">=3.12,<3.14"`. Both ends of
that range are exercised in CI via a build matrix.

**ML framework — Keras 3 on TensorFlow 2.20–2.21.** The model is a Conv1D → Bi-GRU → Dense
regressor built with the Keras functional/sequential API. TensorFlow imports are deliberately
confined to `model`, `training`, `plots`, `pipeline` and `predict`; the remaining modules
(`config`, `data`, `preprocess`, `metrics`, `session`, `errors`) are TensorFlow-free by contract.

**UI framework — Streamlit 1.60 (incoming).** Declared as a `dashboard` dependency group and
resolved in `uv.lock`, but both edits are uncommitted in the working tree and no code imports it.

**Build tool — hatchling.** PEP 517 backend, packaging `src/mlpp` into a wheel. Two console
entrypoints, `mlpp-train` and `mlpp-predict`.

**Test runner — pytest 8.x** with `pytest-cov`, `--strict-markers`, a custom `slow` marker for
TensorFlow-touching tests, and `filterwarnings = ["error::DeprecationWarning:mlpp.*"]` — which
turns the project's own deprecation warnings into failures.

**Package manager — uv**, with `apps/backend/uv.lock` committed. CI syncs with `--locked`, so a
lockfile stale against `pyproject.toml` fails the build rather than silently re-resolving.

**CI — GitHub Actions**, `.github/workflows/ci.yml`. A `quality` job across Python 3.12 and 3.13
running `ruff check`, `ruff format --check`, `mypy src/mlpp` and the full `pytest` suite; plus a
`notebook-drift` job that verifies the committed notebook still matches its generator.

**Deployment — none.** No `Dockerfile`, no platform configuration. This is a deliberate product
decision, restated as a non-goal in the PRD, not a gap.

## Quality Gate Assessment

| Component | Typed | Convention | Training Data | Documented | Verdict |
|---|---|---|---|---|---|
| Language — Python 3.12–3.13 | ✓ | — | ✓ | ✓ | pass |
| ML framework — Keras 3 / TF | — | ✓ | ~ | ✓ | pass with note |
| Build & packaging — uv + hatchling | — | ✓ | ~ | ✓ | pass with note |
| Test runner — pytest | — | — | ✓ | ✓ | pass |
| UI framework — Streamlit (incoming) | — | ✗ | ✓ | ✓ | fail on conventions |

Legend: ✓ = pass, ✗ = fail, ~ = partial, — = not applicable

**11 of 14 applicable criteria pass outright; 2 are partial; 1 fails.**

### Gate Details

#### Type safety — PASS (strongest area)

Python is not typed by default, so this criterion normally fails for Python projects. It passes here
on three independent pieces of evidence:

- `apps/backend/pyproject.toml` `[tool.mypy]` sets `strict = true`, `warn_unreachable = true`,
  `python_version = "3.12"`.
- `[tool.ruff.lint]` `select` includes `"ANN"` (flake8-annotations), making missing annotations a
  lint error rather than a style preference. `[tool.ruff.lint.per-file-ignores]` relaxes this for
  `tests/*` only.
- `.github/workflows/ci.yml` runs `mypy src/mlpp` as a required step on every push and pull request
  to `main`, across both supported Python versions.

Type discipline is therefore mechanically enforced, not merely aspirational. This is a materially
better position than most Python codebases, and it is the single biggest reason an agent can
navigate this repository reliably.

**Minor gap:** there is no `py.typed` marker in `src/mlpp/`. This only affects downstream consumers
importing the built wheel — irrelevant for a single-user local project, but it means the package's
types are invisible to anything installing it as a dependency.

#### Conventions — PASS for the existing package, FAIL for the incoming UI

No web framework governs the existing package, so framework-supplied conventions do not apply.
Under the criteria this normally scores as a partial pass at best. It scores a full pass here
because the project supplies its own conventions and enforces them:

- `CLAUDE.md` documents the directory layout, the one-test-module-per-source-module rule, the
  TensorFlow-free / TensorFlow-owning module split, and four domain invariants (fit-once, feature
  order via `preprocessor.index_of`, single-owner session artifacts, reject-don't-migrate artifacts).
- Those conventions are backed by machinery, not just prose: `--strict-markers` and the `slow`
  marker enforce the fast/slow test split; the `notebook-drift` CI job enforces that the notebook is
  generated rather than hand-edited; `session.py` centralizes every artifact filename in one module.
- The invariants document their own rationale — the session-ownership rule records that three
  modules once drifted and the committed reference run silently stopped loading. Conventions that
  carry their failure history are far more likely to be respected by an agent than bare rules.

**Streamlit fails this criterion.** It has a strong opinion about its execution model (top-to-bottom
script rerun on every interaction, with explicit caching decorators) but almost no opinion about
file layout beyond a `pages/` directory for multipage apps. Two Streamlit projects of any size look
nothing alike. Because this codebase's whole navigability rests on documented conventions, adding a
component that supplies none — with no conventions written for it yet — is the one real gap this
assessment finds.

#### Popularity in training data — PASS, with two version-skew caveats

Assessed within the Python ecosystem, not against JavaScript volume.

- **Python, pytest, Streamlit** — top tier. Abundant idiomatic training data.
- **Keras 3 — partial.** Keras itself is thoroughly represented, but Keras 3 is the multi-backend
  rewrite, and the overwhelming majority of Keras material in any training corpus describes Keras 2
  and `tf.keras`. Expect an agent to reach for `tf.keras.*` imports, `input_shape=` on the first
  layer instead of an explicit `Input`, and the older saving API. The codebase already anticipates
  one instance of this: `model.py:20` carries a comment explaining why an explicit `Input` layer is
  used rather than `input_shape=`.
- **uv — partial.** Now mainstream, but recent enough that pip and Poetry dominate the corpus. An
  agent's default reflex is `pip install`. `CLAUDE.md` already compensates explicitly ("**uv only**
  — never pip/poetry/conda"), which is exactly the right response and demonstrates the compensation
  pattern already works in this repository.
- **hatchling** — adequate. A standard PEP 517 backend, and rarely touched after initial setup.

#### Documentation — PASS across the board

Keras 3 (`keras.io`), TensorFlow, Streamlit, pytest, ruff, mypy and uv (`docs.astral.sh`) all
publish current, versioned, URL-addressable official documentation. No component depends on
outdated or community-maintained material.

## Gaps & Compensation

### Gap 1 — Both instruction files currently forbid the change being planned

**Severity: high.** This is the most consequential finding, and it is not a stack property — it is a
live contradiction between the instruction files and the approved PRD.

- `CLAUDE.md` states: *"**Python only — there is no frontend and no infra stack.**"* and *"Add
  `apps/frontend/` or `infrastructure/` only when something real goes in them."*
- `.cursorrules` line 8 states: *"LAYOUT: apps/backend (Python `mlpp` pkg, src layout) only — no
  frontend/infra here."*

An agent reading either file while implementing the dashboard receives a direct instruction against
the work. Because `.cursorrules` is a dense, literal, "obey literally" contract, it is especially
likely to be followed to the letter. Both files must be updated before implementation starts, or
every agent session will re-litigate whether the dashboard should exist.

Note also that `.cursorrules` describes a general multi-language contract (TypeScript, Python, Go)
while `CLAUDE.md` is project-specific. They overlap on Python and uv rules. That duplication is a
standing drift risk independent of this change.

### Gap 2 — Streamlit supplies no layout conventions

**Severity: medium.** Covered in the conventions detail above. The compensation is to write the
conventions Streamlit does not supply, before any interface code is written — matching how the rest
of this codebase already operates.

The PRD's requirements make several of these conventions non-negotiable rather than stylistic:
FR-011 requires importance logic to be importable independently of the interface, FR-020 requires
artifacts to be resolved only through manifest roles, and FR-019 requires the fast test suite to
stay TensorFlow-free. Streamlit's rerun-everything execution model works against all three unless
the boundary is documented explicitly.

### Gap 3 — Keras 3 version skew

**Severity: low-medium.** Covered above. Compensation is a short API-surface rule.

### Gap 4 — No `py.typed` marker

**Severity: low.** One empty file plus one `pyproject.toml` line. Only matters if the package is
ever consumed as a dependency; noted for completeness.

### Recommended Instruction File Additions

Ready to paste into `CLAUDE.md`.

**Replace the "Python only — there is no frontend" line in `## What this is`:**

```markdown
Python only — no JavaScript frontend and no infra stack. The dashboard (`mlpp.dashboard`) is a
Streamlit app and is still Python; it lives inside the backend package, not in `apps/frontend/`.
```

**Add to `## Project boundaries` → Layout:**

```markdown
  - `apps/backend/src/mlpp/dashboard/` — the Streamlit app. Sits on the **TensorFlow-owning** side
    of the module split (it imports `predict`). Never import it from `config`, `data`, `preprocess`,
    `metrics`, `session` or `errors`.
```

**Add a new `## Dashboard conventions` section:**

```markdown
## Dashboard conventions
Streamlit supplies no layout opinions, so these are ours. `mlpp.dashboard` is a thin presentation
layer — it renders, it does not compute.

- **Entrypoint** is `dashboard/app.py`; run it with
  `uv run --group dashboard streamlit run src/mlpp/dashboard/app.py` from `apps/backend/`.
- **One module per panel** under `dashboard/panels/`, mirroring the PRD's surfaces: `session.py`,
  `introspect.py`, `importance.py`, `single.py`, `batch.py`. Each exposes one `render(...)` taking
  an already-loaded session — panels never load artifacts themselves.
- **No domain logic in `dashboard/`.** Anything that computes belongs in a normal `mlpp` module and
  must be importable and testable without Streamlit. Feature importance lives in `mlpp/importance.py`,
  not in a panel.
- **Caching is explicit**: `@st.cache_resource` for the Keras model (unhashable, one per session
  directory), `@st.cache_data` for dataframes and importance results. Cache keys are
  `(session_dir, dataset_path, seed)` — never mutable objects.
- **Read-only.** The dashboard never writes into `OUTPUTS/`. Downloads go through `st.download_button`.
- **Artifacts resolve through manifest roles only** — `manifest.filenames_for(ROLE_*)`. Spelling a
  filename inside `dashboard/` is the same bug `session.py` exists to prevent.
- **Errors**: catch `MlppError` at the panel boundary and render `st.error(str(exc))`. Never let a
  traceback reach the browser.
- Tests for panels are import-and-call, marked `slow` only when they genuinely load Keras.
```

**Add to `## Core tech stack`:**

```markdown
- Keras 3 (not Keras 2 / `tf.keras`). Import `keras` directly, never `tensorflow.keras`. Use an
  explicit `layers.Input(shape=...)` rather than `input_shape=` on the first layer, and
  `keras.saving.load_model` / `.keras` files for persistence. Most Keras material online predates
  Keras 3 — check `keras.io` before copying an idiom.
- Streamlit 1.x, in the `dashboard` dependency group. `uv sync` without that group must still yield
  a working train/predict install.
```

**Add to `## Testing patterns`:**

```markdown
`mlpp.dashboard` and `mlpp.importance` may import TensorFlow. Keep `importance`'s core loop
parameterised over a scorer callable so the fast suite can exercise it with a stub —
only the genuine Keras path needs the `slow` marker.
```

**Update `.cursorrules` line 8** so it stops contradicting the layout:

```
LAYOUT: apps/backend (Python `mlpp` pkg, src layout) only — no JS frontend/infra. Streamlit UI = `mlpp.dashboard`, still Python, still in apps/backend.
```

**Optional, `py.typed`:** create an empty `apps/backend/src/mlpp/py.typed` and add
`[tool.hatch.build.targets.wheel] ... ` include for it, so the package ships its types.

## Summary

**Overall readiness: ready with compensation** — and toward the strong end of that band.

**Key strengths.** Type safety is enforced mechanically through three independent mechanisms
(mypy strict, ruff's annotation rules, and a required CI step) rather than left to discipline. The
project supplies its own conventions where the language and framework do not, documents the failure
that motivated each invariant, and backs them with real machinery — a strict-marker test split, a
notebook-drift gate, and a single-owner module for artifact filenames. Dependency management is
locked and CI fails on a stale lockfile. For a Python project, this is an unusually agent-legible
codebase.

**Key gaps.** One is urgent and has nothing to do with the stack: `CLAUDE.md` and `.cursorrules`
both currently instruct an agent that no frontend may exist, which directly contradicts the approved
PRD. Fix that before implementation begins. The second is that Streamlit contributes no layout
conventions to a codebase whose navigability depends on them — so the conventions must be written
in advance, as the paste-ready block above does. The third is Keras 3's version skew against a
training corpus dominated by Keras 2, warranting a short API-surface rule.

None of these argue for changing the stack. All three are instruction-file work, which is precisely
where this repository has already shown the pattern works — the existing "uv only, never
pip/poetry/conda" rule is exactly this kind of compensation, and it is effective.

**Recommended next step:** `/10x-health-check`, to audit dependency health, test-suite coverage and
CI completeness against these findings.

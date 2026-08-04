# CLAUDE.md — ml-prediction-pipeline

> Project context for Claude Code. Read fully before editing. Keep edits token-economical.

## What this is
Time-series / tabular **regression** pipeline: Conv1D → Bi-GRU → Dense (Keras 3 on TensorFlow).
Trains sequentially over `TRAIN/*.csv`, evaluates against `TEST/*.csv`, writes a timestamped
artifact directory under `OUTPUTS/`. **Python only — no JavaScript frontend and no infra stack.**
The dashboard (`mlpp.dashboard`) is a Streamlit app and is still Python; it lives inside the backend
package, not in `apps/frontend/`.

## Project boundaries
- Lives on the Kingston SSD under `Projects/ml-prediction-pipeline`. ALL caches/toolchains stay
  project-local (see `.envrc`); never write to $HOME or the internal disk.
- Layout:
  - `apps/backend/` — the `mlpp` uv project (src layout + tests). All logic lives here.
  - `apps/backend/src/mlpp/dashboard/` — the Streamlit app. Sits on the **TensorFlow-owning** side
    of the module split (it imports `predict`). Never import it from `config`, `data`, `preprocess`,
    `metrics`, `session` or `errors`.
  - `notebooks/01_exploration_and_baseline.ipynb` — demonstration report reading the committed
    `OUTPUTS/example/` session; **never** re-add pipeline logic to it and never hand-edit the
    `.ipynb`. Its **outputs are committed** so it renders on GitHub, so regenerate with
    `uv run --project apps/backend python scripts/build_notebook.py --execute` — the bare command
    strips the outputs and the drift gate (source-only) will not warn you.
  - `TRAIN/`, `TEST/` — input CSVs. `OUTPUTS/` — run artifacts (git-ignored except `example/`).
  - `scripts/` — repo-level helpers.
- Add `apps/frontend/` or `infrastructure/` only when something real goes in them.

## Core tech stack
- Python 3.12+ (**uv only** — never pip/poetry/conda). TensorFlow 2.20–2.21, Keras 3, numpy 2.x,
  pandas 3.x, scikit-learn 1.9. Lockfile is `apps/backend/uv.lock`; commit it.
- Keras 3 (not Keras 2 / `tf.keras`). Import `keras` directly, never `tensorflow.keras`. Use an
  explicit `layers.Input(shape=...)` rather than `input_shape=` on the first layer, and
  `keras.saving.load_model` / `.keras` files for persistence. Most Keras material online predates
  Keras 3 — check `keras.io` before copying an idiom.
- Streamlit 1.x, in the `dashboard` dependency group. `uv sync` without that group must still yield
  a working train/predict install.
- TypeScript (strict) applies only if a JavaScript frontend is ever added.

## Build & run
All commands from `apps/backend/` unless noted.

```bash
uv sync --all-groups                 # runtime + dev + notebook groups
uv run mlpp-train --epochs 5         # train + evaluate; --help lists every knob
uv run mlpp-train --no-plots --quiet # CI-friendly run
./scripts/quickstart.sh              # (from repo root) sync + train in one step

# score a new CSV against a trained session (needs TensorFlow — it loads the model)
uv run mlpp-predict --session ../../OUTPUTS/example --input ../../TEST/sample_test_A.csv \
  --output /tmp/preds.csv

# local dashboard (session introspection, feature importance, single-row + batch inference)
uv run --group dashboard streamlit run src/mlpp/dashboard/app.py
```

Notebook: `uv run python -m ipykernel install --user --name mlpp`, then open
`notebooks/01_exploration_and_baseline.ipynb` and select the `mlpp` kernel.

## Testing patterns
One test module per source module, in `apps/backend/tests/` mirroring `src/mlpp/`.
Run from `apps/backend/` so `pyproject.toml` is picked up as the pytest configfile.

```bash
uv run pytest -m 'not slow'   # ~1s, no TensorFlow import — use this while iterating
uv run pytest                 # full suite incl. Keras smoke tests (~30s)
uv run ruff check . && uv run ruff format --check .
uv run mypy src/mlpp          # strict; must stay clean
```

Tests touching Keras/TF are marked `@pytest.mark.slow`. Keep `config`, `data`, `preprocess`,
`metrics`, `session`, `errors` and `importance` **free of TensorFlow imports** — that separation is
what keeps the fast suite fast, and it is what lets `load_session` validate a session without the
training stack. `model`, `training`, `plots`, `pipeline`, `predict` and `dashboard` own the heavy
imports. Run tests before declaring done.

`importance`'s core loop is parameterised over a scorer callable so the fast suite can exercise it
with a stub — only the genuine Keras path needs the `slow` marker. Tests for panels are
import-and-call, marked `slow` only when they genuinely load Keras.

## Domain invariants
- **Fit once.** `Preprocessor.fit` runs on the first TRAIN file only; every later file and all TEST
  data go through `transform`. Re-fitting on test data leaks and silently inflates R².
- **Feature order is not config order.** After one-hot expansion, axis 1 of `X` follows
  `Preprocessor.feature_names`. Locate a column with `preprocessor.index_of(name)` — never with
  `input_columns.index(name)`.
- **One owner for session artifacts.** `session.py` names every file in a session directory;
  no other module may spell one out. Write through `SessionWriter.register(role, filename)` and
  use the path it returns. `manifest.json` is the single source of truth for the column
  contract — never persist feature names or the schema anywhere else. Three modules once did,
  they drifted, and the committed reference run silently stopped loading.
- **Artifacts are rejected, not migrated.** A session whose `schema_version` differs raises
  `SchemaVersionError`. Bump `SCHEMA_VERSION` on a breaking layout change; do not add
  compatibility shims — regenerating a session takes seconds.
- `X` is shaped `(rows, n_features, 1)`; Conv1D runs across the feature axis, not time.
- Errors are explicit: raise from the `MlppError` hierarchy in `errors.py`, never return `None`
  to signal failure. The one exception is `transform`, which returns `y=None` when the target
  column is genuinely absent (inference input).

## Dashboard conventions
Streamlit supplies no layout opinions, so these are ours. `mlpp.dashboard` is a thin presentation
layer — it renders, it does not compute.

- **Entrypoint** is `dashboard/app.py`; run it with
  `uv run --group dashboard streamlit run src/mlpp/dashboard/app.py` from `apps/backend/`.
- **One module per panel** under `dashboard/panels/`, mirroring the PRD's surfaces: `introspect.py`,
  `importance.py`, `single.py`, `batch.py`. Each exposes one `render(...)` taking an already-loaded
  session — panels never load artifacts themselves.
- **No domain logic in `dashboard/`.** Anything that computes belongs in a normal `mlpp` module and
  must be importable and testable without Streamlit. Feature importance lives in `mlpp/importance.py`,
  not in a panel.
- **Caching is explicit**: `@st.cache_resource` for the Keras model (unhashable, one per session
  directory), `@st.cache_data` for dataframes and importance results. Cache keys are scalars —
  `(session_dir, dataset_path, seed, n_repeats)` — never mutable objects.
- **Read-only.** The dashboard never writes into `OUTPUTS/`. Downloads go through `st.download_button`.
- **Artifacts resolve through manifest roles only** — `manifest.filenames_for(ROLE_*)`. Spelling a
  filename inside `dashboard/` is the same bug `session.py` exists to prevent.
- **Errors**: catch `MlppError` at the panel boundary and render `st.error(str(exc))`. Never let a
  traceback reach the browser.

## Working agreement
- Smallest correct change; isolated, pure modules. Explicit error handling. No `any`.
- Conventional Commits. Reason in English; minimize tokens to avoid context rot.

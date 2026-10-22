# CLAUDE.md — ml-prediction-pipeline

> Project context for Claude Code. Read fully before editing. Keep edits token-economical.

## What this is
Time-series / tabular **regression** MLOps pipeline: Conv1D → Bi-GRU → Dense, trained in **PyTorch**,
exported to **ONNX** and served by **ONNX Runtime** behind an async **FastAPI** service. Trains
sequentially over `TRAIN/*.csv`, evaluates against `TEST/*.csv`, writes a timestamped session under
`OUTPUTS/`, records the run in **MLflow**, and captures served payloads for **Evidently** drift checks.
**Python only — no JavaScript frontend.** Infra is limited to `docker/` + `docker-compose.yml`
(API + MLflow server). The dashboard (`mlpp.dashboard`) is a Streamlit app inside the backend package.

## Project boundaries
- Lives on the Kingston SSD under `Projects/ml-prediction-pipeline`. ALL caches/toolchains stay
  project-local (see `.envrc`); never write to $HOME or the internal disk.
- Layout:
  - `apps/backend/` — the `mlpp` uv project (src layout + tests). All logic lives here.
  - `apps/backend/src/mlpp/` subpackages: `schemas/` (Pydantic wire models), `training/` (torch:
    model, loop, export, tracking), `inference/` (ONNX Runtime + FastAPI), `monitoring/` (payload
    capture, drift). `dashboard/` is the Streamlit app; it imports `predict` (ONNX Runtime). Never
    import it from `config`, `data`, `preprocess`, `metrics`, `session` or `errors`.
  - `docker/api.Dockerfile` — serving image, build context `apps/backend`; `docker-compose.yml` at
    the root runs `api` (:8100) and `mlflow` (:5050), both bound to 127.0.0.1.
  - `notebooks/01_exploration_and_baseline.ipynb` — demonstration report reading the committed
    `OUTPUTS/example/` session; **never** re-add pipeline logic to it and never hand-edit the
    `.ipynb`. Its **outputs are committed** so it renders on GitHub, so regenerate with
    `uv run --project apps/backend python scripts/build_notebook.py --execute` — the bare command
    strips the outputs and the drift gate (source-only) will not warn you.
  - `TRAIN/`, `TEST/` — input CSVs. `OUTPUTS/` — run artifacts (git-ignored except `example/`).
    `PAYLOADS/` — captured inference payloads; `mlruns/` — local MLflow store (both git-ignored).
  - `scripts/` — repo-level helpers.
- Add `apps/frontend/` only when something real goes in it.

## Core tech stack
- Python 3.12+ (**uv only** — never pip/poetry/conda). numpy 2.x, pandas 3.x, scikit-learn 1.9.
  Lockfile is `apps/backend/uv.lock`; commit it.
- Dependency split — keep it: core (`onnxruntime`, `pydantic`) scores sessions; extras `train`
  (torch, onnx, onnxscript, mlflow, matplotlib, plotly), `serve` (fastapi, uvicorn,
  pydantic-settings), `monitor` (evidently). The serving image installs `--extra serve` only and its
  build fails if torch/mlflow are importable — never import them from `schemas`, `inference`,
  `predict`, `predict_cli` or `monitoring.payloads`.
- PyTorch 2.x. ONNX export uses the `torch.export` exporter (`dynamo=True`, opset 18, dynamic batch
  axis, fixed feature axis) and must pass the ONNX Runtime parity check in `training/export.py`.
  Load checkpoints with `torch.load(..., weights_only=True)` only.
- MLflow 3.x. Tracking lives behind `training/tracking.Tracker`; nothing else imports `mlflow`.
  Logged models use `name=` (LoggedModel entities), not run artifact paths.
- Evidently 0.7 (`from evidently import Report, Dataset, DataDefinition`). Imported lazily inside
  `monitoring/drift.detect_drift`; the 0.4-era `ColumnMapping` / `Report(metrics=[...])` API is gone.
- Streamlit 1.x, in the `dashboard` dependency group.
- TypeScript (strict) applies only if a JavaScript frontend is ever added.

## Build & run
All commands from `apps/backend/` unless noted.

```bash
uv sync --all-extras --all-groups    # runtime + train/serve/monitor extras + dev/notebook/dashboard
uv run mlpp-train --epochs 5         # train + evaluate + MLflow (repo-local mlruns/ by default)
uv run mlpp-train --no-plots --quiet --no-tracking   # CI-friendly run
./scripts/quickstart.sh              # (from repo root) sync + train in one step

# score a CSV against a trained session (ONNX Runtime; no torch needed)
uv run mlpp-predict --session ../../OUTPUTS/example --input ../../TEST/sample_test_A.csv \
  --output /tmp/preds.csv

uv run mlpp-serve --session ../../OUTPUTS/example                   # API on 127.0.0.1:8100
uv run mlpp-drift --session ../../OUTPUTS/example --current ../../PAYLOADS   # exit 2 = drift
docker compose up -d --build         # (from repo root) api :8100 + mlflow :5050

# local dashboard (session introspection, feature importance, single-row + batch inference)
uv run --group dashboard streamlit run src/mlpp/dashboard/app.py
```

Notebook: `uv run python -m ipykernel install --user --name mlpp`, then open
`notebooks/01_exploration_and_baseline.ipynb` and select the `mlpp` kernel.

## Testing patterns
One test module per source module, in `apps/backend/tests/` mirroring `src/mlpp/`.
Run from `apps/backend/` so `pyproject.toml` is picked up as the pytest configfile.

```bash
uv run pytest -m 'not slow'   # seconds; no torch, no Evidently, no MLflow runs — use while iterating
uv run pytest                 # full suite: trains, exports, tracks, runs Evidently (~2 min)
uv run ruff check . && uv run ruff format --check .
uv run mypy src/mlpp          # strict; must stay clean
```

Tests that train/export with torch, open real MLflow runs or run Evidently are marked
`@pytest.mark.slow`. Keep `config`, `data`, `preprocess`, `metrics`, `session`, `errors`,
`importance` and `schemas` **free of model-runtime imports** — that is what keeps the fast suite fast
and lets `load_session` validate a session without any runtime. `training/*`, `plots` and `pipeline`
own torch; `predict`, `inference` and `dashboard` own ONNX Runtime. Run tests before declaring done.

API tests (`tests/test_api.py`) run the real app and mock only the ONNX `InferenceSession` via
`conftest.make_mock_ort_session`; `conftest.write_linear_onnx` builds a tiny real graph (no torch)
for genuine ONNX Runtime coverage in the fast suite. `importance`'s loop takes a scorer callable so
the fast suite can use a stub.

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
  compatibility shims — regenerating a session takes seconds. v2 = ONNX `best_model` + torch `.pt`
  checkpoints; regenerate `OUTPUTS/example` (and re-execute the notebook) after a bump.
- **One scoring path.** API, `mlpp-predict`, dashboard and notebook all score through
  `mlpp.predict.score_frame` on the session's ONNX graph. Never score with the in-memory torch
  model outside training/evaluation.
- **The request schema is derived, never written.** `schemas/inference.build_request_model` builds
  it from the manifest's `FeatureContract`. Strict by design: required, non-null, finite, no extra
  keys — do not loosen it to match the preprocessor's training-time leniency.
- **Payload capture is best-effort.** It is the one place an error is logged and not raised: a
  monitoring failure must never fail a prediction. Everything else stays explicit.
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
- **Caching is explicit**: `@st.cache_resource` for the ONNX model (unhashable, one per session
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

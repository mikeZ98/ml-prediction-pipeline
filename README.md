# ML Prediction Pipeline — Time-Series Regression (CNN + Bi-GRU)

End-to-end pipeline for **time-series / tabular regression**:

- **Model**: Conv1D → Bidirectional GRU → Dense (Keras 3 / TensorFlow)
- **Preprocessing**: `StandardScaler` for inputs and target, optional one-hot for categoricals
- **Artifacts**: model (`.keras`), scalers/encoders (`.gz`), and a versioned `manifest.json`
- **Evaluation**: MSE/RMSE/MAE/R², learning curves, interactive HTML diff plots
- **Interfaces**: a `mlpp-train` CLI *and* a thin notebook driver over the same tested package

---

## 📦 Repository structure

```
.
├─ apps/backend/                 # the `mlpp` uv project — all pipeline logic
│  ├─ src/mlpp/                  #   config · data · preprocess · metrics · session
│  │                             #   model · training · plots · pipeline · cli
│  ├─ tests/                     #   one test module per source module
│  ├─ pyproject.toml             #   deps, pytest/ruff/mypy config
│  └─ uv.lock                    #   pinned, committed
├─ notebooks/01_train.ipynb      # thin driver over `mlpp` (generated — see below)
├─ TRAIN/                        # training CSVs (samples included)
├─ TEST/                         # test CSVs (samples included)
├─ OUTPUTS/                      # run artifacts, git-ignored except example/
│  └─ example/                   # a committed reference run
├─ scripts/
│  ├─ quickstart.sh              # sync + train in one command
│  └─ build_notebook.py          # regenerates notebooks/01_train.ipynb
├─ CLAUDE.md · .cursorrules      # agent context
├─ LICENSE
└─ README.md
```

---

## 🚀 Quickstart

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). **No pip, venv or conda.**

```bash
./scripts/quickstart.sh              # sync dependencies, then train
./scripts/quickstart.sh --epochs 20  # extra flags pass through to mlpp-train
```

Or step by step, from `apps/backend/`:

```bash
uv sync --all-groups
uv run mlpp-train --epochs 5
uv run mlpp-train --help     # every knob: columns, loss, seed, batch size, plots…
```

Artifacts land in a timestamped folder under `OUTPUTS/`.

### Notebook

```bash
cd apps/backend && uv run python -m ipykernel install --user --name mlpp
```

Open `notebooks/01_train.ipynb` and pick the `mlpp` kernel. The notebook only builds a config and
calls `run_pipeline` — the logic lives in the package, so it stays testable. It is generated:

```bash
uv run --project apps/backend python scripts/build_notebook.py
```

---

## 🧠 Task & data schema

Regression on time-ordered rows: predict a continuous `target`.

Default input columns — override with `--input-columns`:

```
feature_01 .. feature_12, comp_active
```

CSV delimiters are sniffed, with `;` and `,` fallbacks. Under the default `strict_schema`, a
missing input column fails fast; `--lenient-schema` fills it with `0.0` instead.

---

## 📊 Outputs

Per session directory `OUTPUTS/<timestamp>/`:

| File | Contents |
| --- | --- |
| `manifest.json` | schema version, the column contract, and an inventory of every file below |
| `best_model.keras` | best checkpoint by `val_loss` |
| `model_iter_NN.keras` | snapshot after each training stage |
| `scaler.gz`, `output_scaler.gz`, `encoders.gz` | fitted preprocessing estimators |
| `history_train_NN.csv`, `training_log.csv` | per-epoch metrics |
| `training_curves_train_NN.png` | loss / MAE curves |
| `prediction_analysis_trNN_teNN.html` | interactive truth-vs-prediction + residuals |
| `test_metrics.csv` | MSE/RMSE/MAE/R² per (train stage, test file) pair |

`manifest.json` is the single source of truth for the column contract, and
`mlpp.session.load_session()` reads it back. It carries a `schema_version`:
sessions written under a different version are **rejected, not migrated** —
regenerate them with `mlpp-train`. Artifacts take seconds to rebuild, so a
permanent compatibility shim would cost more than it saves.

---

## 🧪 Development

From `apps/backend/`:

```bash
uv run pytest -m 'not slow'   # ~1s — TF-free modules only
uv run pytest                 # full suite, incl. Keras smoke tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src/mlpp          # strict
```

`config`, `data`, `preprocess`, `metrics` and `session` deliberately import no TensorFlow, which
is what keeps the fast suite fast — and is why reading a saved session never needs the training stack.

---

## 🔄 Reproducibility & correctness

- Seeded Python/NumPy/Keras RNGs (`--seed`), verified reproducible by test.
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint + CSVLogger.
- **Preprocessing is fitted once**, on the first training file; test data is only ever
  transformed, so evaluation cannot leak.
- Feature order is frozen at fit time and persisted, so a dtype change in a later CSV can never
  silently reorder the model's inputs.

---

## 🤖 CI

`.github/workflows/ci.yml` runs on every push and PR to `main`:

- **quality** (Python 3.12 and 3.13) — `uv sync --locked`, then `ruff check`, `ruff format --check`,
  `mypy --strict`, and the full `pytest` suite. `--locked` fails if `uv.lock` is stale against
  `pyproject.toml` instead of silently re-resolving.
- **notebook-drift** — regenerates `notebooks/01_train.ipynb` and fails if it differs, so the
  generated notebook can never drift from `scripts/build_notebook.py`.

## 🗺️ Roadmap

- [ ] `mlpp-predict` CLI: load a session directory, score a new CSV
- [x] GitHub Actions: lint, types and tests on push
- [ ] MLflow experiment tracking (optional)

---

## 📄 License

MIT — see [LICENSE](LICENSE).

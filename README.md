# ML Prediction Pipeline — Time-Series Regression (CNN + Bi-GRU)

End-to-end pipeline for **time-series / tabular regression**:

- **Model**: Conv1D → Bidirectional GRU → Dense (Keras 3 / TensorFlow)
- **Preprocessing**: `StandardScaler` for inputs and target, optional one-hot for categoricals
- **Artifacts**: model (`.keras`), scalers/encoders (`.gz`), and a versioned `manifest.json`
- **Evaluation**: MSE/RMSE/MAE/R², learning curves, interactive HTML diff plots
- **Interfaces**: `mlpp-train` and `mlpp-predict` CLIs *and* a thin notebook driver over the same
  tested package

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
├─ notebooks/                    # demo report over OUTPUTS/example (generated — see below)
│  └─ 01_exploration_and_baseline.ipynb
├─ TRAIN/                        # training CSVs (samples included)
├─ TEST/                         # test CSVs (samples included)
├─ OUTPUTS/                      # run artifacts, git-ignored except example/
│  └─ example/                   # a committed reference run
├─ scripts/
│  ├─ quickstart.sh              # sync + train in one command
│  ├─ build_notebook.py          # regenerates the notebook from its cell source
│  └─ check_notebook_drift.py    # fails if the notebook's source left the generator
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

### Scoring a new CSV

Once a session exists, score fresh data against it with `mlpp-predict`. The column
contract is read from the session's own `manifest.json`, so you only supply the
session directory and an input file:

```bash
uv run mlpp-predict \
  --session ../../OUTPUTS/example \
  --input ../../TEST/sample_test_A.csv \
  --output /tmp/preds.csv

uv run mlpp-predict --help   # --keep-inputs, --lenient-schema, --quiet
```

Predictions come back in the target's original units and are written to the path
you name — never into the session directory, which stays a reproducible training
artifact. A target column in the input is optional and ignored if present.

Two things worth knowing:

- **Inference requires TensorFlow.** Loading a `.keras` model needs the full stack,
  so this is not a lightweight install. Only *inspecting* a session is cheap:
  `load_session()` validates the manifest and restores the preprocessor without
  importing Keras at all.
- **Unseen categorical levels warn rather than fail.** A category absent from
  training is encoded as all-zeros (the encoder is built with
  `handle_unknown="ignore"`), which still yields a number. The command reports the
  affected columns and row count on stderr and exits `0`, so one stray value cannot
  kill a batch — but you should treat those rows as suspect.

### Notebook — demonstration report

`notebooks/01_exploration_and_baseline.ipynb` is a **report, not an exercise**: it reads the
committed `OUTPUTS/example/` session and shows the feature contract, the recorded baseline metrics,
freshly scored predictions in the target's own units, and the interactive truth-vs-prediction plot.
It trains nothing, so it is deterministic and needs no data or GPU — and because its outputs are
committed, you can read the whole thing on GitHub without running it.

To run it yourself:

```bash
cd apps/backend && uv run python -m ipykernel install --user --name mlpp
```

Then open the notebook and pick the `mlpp` kernel. All logic lives in the `mlpp` package, so the
cells only call into tested code. It is generated:

```bash
# regenerate the cells only — leaves the notebook output-free
uv run --project apps/backend python scripts/build_notebook.py

# regenerate AND execute, storing outputs — this is what gets committed
uv run --project apps/backend python scripts/build_notebook.py --execute
```

> **Use `--execute` when committing.** Plain regeneration emits an output-free notebook, so
> committing that would leave the report blank on GitHub. CI compares cell *source* only, so
> outputs never fail the drift gate — which also means nothing warns you if you commit it
> output-stripped. `--execute` is the one command that produces the committed artifact.

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
- **notebook-drift** — `scripts/check_notebook_drift.py` compares the committed notebook's cell
  source against `scripts/build_notebook.py`, ignoring `outputs` and `execution_count` so the
  report can carry executed results while its code still comes from the generator. The `quality`
  job additionally *executes* the notebook (`tests/test_notebook.py`) — the drift check alone
  proves only that nobody hand-edited it, not that it still runs.

## 🗺️ Roadmap

- [x] `mlpp-predict` CLI: load a session directory, score a new CSV
- [x] GitHub Actions: lint, types and tests on push
- [ ] MLflow experiment tracking (optional)

---

## 📄 License

MIT — see [LICENSE](LICENSE).

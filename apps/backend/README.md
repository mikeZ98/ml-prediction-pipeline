# mlpp — backend

Training/evaluation pipeline for time-series & tabular regression (Conv1D → Bi-GRU → Dense, Keras 3).

```bash
uv sync --all-groups          # install runtime + dev + notebook groups
uv run mlpp-train --epochs 5  # train against ../../TRAIN and ../../TEST
uv run pytest -q -m 'not slow'  # fast suite (no TensorFlow import)
uv run pytest -q                # full suite
```

## Module map

| Module | Responsibility | Imports TF? |
| --- | --- | --- |
| `config.py` | Immutable `DataConfig` / `TrainConfig` / `EvalConfig` | no |
| `data.py` | CSV discovery + delimiter-sniffing loader | no |
| `preprocess.py` | Feature schema, scaling, one-hot (no file I/O) | no |
| `metrics.py` | MSE/RMSE/MAE/R², rolling residuals | no |
| `session.py` | Owns every session-dir filename, the versioned manifest, load/save | no |
| `model.py` | Keras model construction, GPU memory growth | yes |
| `training.py` | Seeding, callbacks, sample weights, `fit` | yes |
| `plots.py` | PNG training curves, interactive HTML reports | yes |
| `pipeline.py` | Orchestration across TRAIN/TEST files | yes |
| `cli.py` | `mlpp-train` entrypoint | yes |

The TF-free modules hold the logic worth unit-testing; `pytest -m 'not slow'` covers them in
about a second because TensorFlow is never imported.

## Feature-order contract

`Preprocessor` resolves the numeric/categorical split **once**, during `fit`, and every later
`transform` reuses it. Locate a column inside `X` with `preprocessor.index_of(name)` — the raw
config order is not the axis-1 order once one-hot encoding expands categoricals.

## Session-artifact contract

`session.py` is the only module allowed to name a file in a session directory. Everything else
registers with `SessionWriter` and gets a path back. `manifest.json` records the column contract
exactly once plus an inventory of every file written, and `load_session()` reads it back,
verifying that what the manifest claims is actually on disk.

Sessions carry a `SCHEMA_VERSION`; mismatches raise `SchemaVersionError` rather than being
migrated. If you add an artifact, register it with a role — an unregistered file is invisible
to every reader.

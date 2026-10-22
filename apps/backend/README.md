# mlpp — backend

Training, serving and monitoring for time-series & tabular regression (Conv1D → Bi-GRU → Dense):
PyTorch for training, ONNX Runtime for every scorer.

```bash
uv sync --all-extras --all-groups   # runtime + train/serve/monitor extras + dev groups
uv run mlpp-train --epochs 5        # train against ../../TRAIN and ../../TEST, tracked in MLflow
uv run mlpp-serve --session ../../OUTPUTS/example
uv run mlpp-drift --session ../../OUTPUTS/example --current ../../PAYLOADS
uv run pytest -q -m 'not slow'      # fast suite
uv run pytest -q                    # full suite
```

## Module map

| Module | Responsibility | Heavy imports |
| --- | --- | --- |
| `config.py` | Immutable `DataConfig` / `TrainConfig` / `EvalConfig` / `TrackingConfig` | — |
| `data.py` | CSV/Parquet discovery and delimiter-sniffing loader | — |
| `preprocess.py` | Feature schema, scaling, one-hot (no file I/O) | — |
| `metrics.py` | MSE/RMSE/MAE/R², rolling residuals | — |
| `session.py` | Owns every session-dir filename, the versioned manifest, load/save | — |
| `importance.py` | Permutation importance over an injected scorer | — |
| `schemas/inference.py` | Strict Pydantic v2 wire models, built from the manifest | pydantic |
| `inference/onnx_model.py` | `OnnxModel`: validated ONNX Runtime execution | onnxruntime |
| `predict.py` / `predict_cli.py` | Score a frame against a session; `mlpp-predict` | onnxruntime |
| `inference/service.py`, `api.py`, `settings.py` | Async FastAPI service; `mlpp-serve` | `serve` extra |
| `monitoring/payloads.py` | JSONL payload capture and reading | — |
| `monitoring/drift.py`, `drift_cli.py` | Evidently drift detection; `mlpp-drift` | `monitor` extra (lazy) |
| `training/model.py` | The PyTorch network, loss factory | torch |
| `training/loop.py` | Fit loop: seeding, weights, early stop, LR plateau, checkpoints | torch |
| `training/export.py` | ONNX export with an ONNX Runtime parity check | torch, onnx |
| `training/tracking.py` | `Tracker` protocol, MLflow implementation (lazy) | `train` extra (lazy) |
| `pipeline.py`, `cli.py`, `plots.py` | Orchestration across TRAIN/TEST; `mlpp-train` | torch |
| `dashboard/` | Streamlit session explorer | streamlit, onnxruntime |

Nothing on the serving path (`schemas`, `inference`, `predict`, `monitoring.payloads`) imports
torch, mlflow or Evidently. The Docker build asserts this.

## Feature-order contract

`Preprocessor` resolves the numeric/categorical split **once**, during `fit`, and every later
`transform` reuses it. Locate a column inside `X` with `preprocessor.index_of(name)`. Once one-hot
expansion runs, the raw config order no longer matches the axis-1 order.

## Session-artifact contract

`session.py` is the only module allowed to name a file in a session directory. Everything else
registers with `SessionWriter` and gets a path back. `manifest.json` records the column contract
exactly once, plus an inventory of every file written. `load_session()` reads it back and checks
that what the manifest claims is actually on disk. Scorers resolve the model through the
`best_model` role, and `load_onnx_model` cross-checks the graph's feature width against the
manifest.

Sessions carry a `SCHEMA_VERSION` (currently 3); a mismatch raises `SchemaVersionError` rather
than being migrated. If you add an artifact, register it with a role, or no reader will see it.

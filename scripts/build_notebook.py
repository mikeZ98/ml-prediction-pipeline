#!/usr/bin/env python3
"""Regenerate notebooks/01_train.ipynb from the cell definitions below.

The notebook is a thin driver over the `mlpp` package — keeping its source here
means the pipeline logic lives in one place (apps/backend/src/mlpp) and the
notebook JSON never drifts from it.

    uv run --project apps/backend python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "01_train.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Training pipeline

Thin driver over the `mlpp` package (`apps/backend/src/mlpp`). All logic —
preprocessing, model, training, evaluation — lives there and is unit-tested;
this notebook only chooses the configuration and inspects the results.

Setup (once, from the repo root):

```bash
uv sync --project apps/backend --all-groups
uv run --project apps/backend python -m ipykernel install --user --name mlpp
```

Prefer a one-liner? `uv run --project apps/backend mlpp-train --epochs 5`
""",
    ),
    (
        "code",
        """import logging
from pathlib import Path

from mlpp.config import ColumnConfig, DataConfig, EvalConfig, PipelineConfig, TrainConfig
from mlpp.model import configure_gpu_memory_growth

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
print("repo root:", REPO_ROOT)
print("GPUs with memory growth enabled:", configure_gpu_memory_growth())""",
    ),
    (
        "markdown",
        """## Configuration

Every knob the old notebook exposed as a module-level global is now a field on
one of four frozen dataclasses. They validate on construction, so a bad value
fails here rather than deep inside training.""",
    ),
    (
        "code",
        """# Which columns matter — the contract a trained session records in its manifest.
columns = ColumnConfig(
    input_columns=(
        "feature_01", "feature_02", "feature_03", "feature_04", "feature_05", "feature_06",
        "feature_07", "feature_08", "feature_09", "feature_10", "feature_11", "feature_12",
        "comp_active",
    ),
    output_column="target",
    categorical_columns=(),   # one-hot these regardless of dtype
    strict_schema=True,       # False -> fill missing input columns with 0.0
)

# Where the data lives. Kept separate from the column contract so a reader that
# scores an existing session needs only `columns`, never these directories.
data = DataConfig(
    train_dir=REPO_ROOT / "TRAIN",
    test_dir=REPO_ROOT / "TEST",
    output_dir=REPO_ROOT / "OUTPUTS",
    columns=columns,
    test_files=(),            # e.g. ("sample_test_A.csv",); empty = glob TEST/
)

training = TrainConfig(
    epochs=5,
    batch_size=128,
    seed=50,
    loss="mse",               # or "huber"
    active_weight_alpha=0.0,  # >0 up-weights rows where comp_active != 0
)

evaluation = EvalConfig(rolling_window=500, write_plots=True)

cfg = PipelineConfig(data=data, train=training, eval=evaluation)
cfg""",
    ),
    (
        "markdown",
        """## Run

Trains on each `TRAIN/*.csv` in turn — reusing the feature space fitted on the
first file — and evaluates against every `TEST/*.csv` after each stage.""",
    ),
    (
        "code",
        """from mlpp.pipeline import run_pipeline

result = run_pipeline(cfg)
print("artifacts ->", result.session_dir)""",
    ),
    (
        "markdown",
        """## Results

`test_metrics.csv`, the training curves and the interactive HTML diff plots are
all written into the timestamped session directory.""",
    ),
    (
        "code",
        """import pandas as pd

pd.DataFrame([row.as_dict() for row in result.rows])""",
    ),
    (
        "code",
        """from IPython.display import IFrame

# Interactive truth-vs-prediction report for train stage 1 / test file 1.
IFrame(str((result.session_dir / "prediction_analysis_tr01_te01.html").relative_to(Path.cwd())),
       width="100%", height=850)""",
    ),
]


def build() -> dict[str, object]:
    cells = []
    for kind, source in CELLS:
        lines = source.strip("\n").splitlines(keepends=True)
        cell: dict[str, object] = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "mlpp",
                "language": "python",
                "name": "mlpp",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK.relative_to(REPO_ROOT)}")

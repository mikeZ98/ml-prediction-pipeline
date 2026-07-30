#!/usr/bin/env python3
"""Regenerate notebooks/01_exploration_and_baseline.ipynb from the cells below.

The notebook is a demonstration report over the committed `OUTPUTS/example/`
session: it reads a trained session's contract, metrics and predictions rather
than training anything. Keeping its source here means the logic lives in one
place (apps/backend/src/mlpp) and the notebook JSON never drifts from it.

The committed notebook carries executed outputs so it renders on GitHub without
being run. Regenerating with this script alone produces an output-free notebook —
re-execute it before committing if you want the outputs refreshed. The CI drift
check compares cell source only, so outputs never fail the gate.

    uv run --project apps/backend python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "01_exploration_and_baseline.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Sequential time-series regression — exploration & baseline

A demonstration report over the committed reference run in `OUTPUTS/example/`.

**The model.** A Conv1D → stacked Bi-GRU → linear head regressor (Keras 3 on
TensorFlow), trained sequentially over `TRAIN/*.csv` and evaluated against every
`TEST/*.csv` after each stage.

**What this notebook does.** It *reads* a trained session — nothing is trained
here. Everything below comes from `OUTPUTS/example/`, a committed artifact
directory, so this report is deterministic and needs no data, no GPU and no
training run to reproduce. All logic lives in the tested `mlpp` package
(`apps/backend/src/mlpp`); the cells only call into it.

To train your own session or score your own CSV, see the two CLIs at the end.
""",
    ),
    (
        "markdown",
        """## The session contract

Every session directory carries a `manifest.json` that is the single source of
truth for what the model expects and what the directory contains. A reader never
has to guess a filename or restate the column list — it asks the manifest.

Note that `load_session` is TensorFlow-free: inspecting and validating a session
is cheap, and the model is only loaded when you actually score something.
""",
    ),
    (
        "code",
        """from pathlib import Path

from mlpp.config import ColumnConfig
from mlpp.session import read_manifest, load_session

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SESSION_DIR = REPO_ROOT / "OUTPUTS" / "example"

# The column contract comes from the manifest, not from a hardcoded list.
features = read_manifest(SESSION_DIR).features
columns = ColumnConfig(
    input_columns=features.numeric_columns + features.categorical_columns,
    output_column=features.output_column,
    categorical_columns=features.categorical_columns,
)

session = load_session(SESSION_DIR, columns)
manifest = session.manifest

print(f"schema version : {manifest.schema_version}")
print(f"trained        : {manifest.created}")
print(f"target         : {features.output_column}")
print(f"inputs         : {len(features.numeric_columns)} numeric, "
      f"{len(features.categorical_columns)} categorical")
print(f"model features : {session.preprocessor.n_features} (after one-hot expansion)")
print(f"artifacts      : {len(manifest.artifacts)} files recorded")""",
    ),
    (
        "markdown",
        """## Baseline accuracy

Recorded at training time, one row per (train stage, test file) pair. The model
trains on each `TRAIN` file in turn while reusing the feature space fitted on the
first one, so later stages show the effect of additional data.
""",
    ),
    (
        "code",
        """import pandas as pd

from mlpp.session import ROLE_METRICS

# Filename resolved through the manifest — session.py owns every name in here.
metrics_file = manifest.filenames_for(ROLE_METRICS)[0]
metrics = pd.read_csv(SESSION_DIR / metrics_file)
metrics.round(4)""",
    ),
    (
        "markdown",
        """## Scoring new data

The same session can score a CSV it has never seen, through the inference seam
(`mlpp.predict`). Predictions come back in the target's original units — the
target scaler is inverted on the way out, so these are directly comparable to the
truth column rather than living in standardised space.

Unseen categorical levels would be reported here rather than silently encoded as
zeros; this input has none.
""",
    ),
    (
        "code",
        """from mlpp.data import read_csv_auto
from mlpp.predict import load_model, score_frame

frame = read_csv_auto(REPO_ROOT / "TEST" / "sample_test_A.csv")
scored = score_frame(session, load_model(session), frame)

comparison = pd.DataFrame({
    "actual": frame[features.output_column],
    "predicted": scored.predictions,
})
comparison["error"] = comparison["predicted"] - comparison["actual"]

print(f"rows scored     : {len(comparison)}")
print(f"unseen levels   : {scored.describe_unseen() or 'none'}")
print()
print(comparison.describe().loc[["mean", "std", "min", "max"]].round(4))
comparison.head()""",
    ),
    (
        "markdown",
        """## Truth vs prediction

The interactive report below is a committed artifact of the reference run — pan
and zoom to inspect where the model tracks the signal and where it drifts.
""",
    ),
    (
        "code",
        """from IPython.display import IFrame

from mlpp.session import ROLE_PREDICTION_ANALYSIS

# An IFrame src is resolved by the browser against this notebook's own directory
# (<repo>/notebooks), which is fixed — so the path is relative to that, never to
# the kernel's cwd. The filename itself comes from the manifest.
report_name = manifest.filenames_for(ROLE_PREDICTION_ANALYSIS)[0]
IFrame(f"../OUTPUTS/example/{report_name}", width="100%", height=850)""",
    ),
    (
        "markdown",
        """## Running it yourself

Everything above reads an existing session. To produce or consume one, from
`apps/backend/`:

```bash
# train a new session into OUTPUTS/<timestamp>/
uv run mlpp-train --epochs 5
uv run mlpp-train --help      # columns, loss, seed, batch size, plots…

# score a CSV against any session directory
uv run mlpp-predict --session ../../OUTPUTS/example \\
  --input ../../TEST/sample_test_A.csv --output preds.csv
```

This notebook is generated from `scripts/build_notebook.py` and must not be
hand-edited; CI checks its source against the generator.
""",
    ),
]


def build() -> dict[str, object]:
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        lines = source.strip("\n").splitlines(keepends=True)
        # nbformat >=4.5 requires a cell id, and warns that omitting one will become
        # a hard error. Derived from the index so regeneration stays byte-identical.
        cell: dict[str, object] = {
            "cell_type": kind,
            "id": f"cell-{index:02d}",
            "metadata": {},
            "source": lines,
        }
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

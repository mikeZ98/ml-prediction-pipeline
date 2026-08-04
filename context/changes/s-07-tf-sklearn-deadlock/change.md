---
change_id: s-07-tf-sklearn-deadlock
title: Unblock the full test suite — TensorFlow/scikit-learn OpenMP deadlock
status: implemented
created: 2026-08-04
updated: 2026-08-04
---

## Notes

`uv run pytest` (full suite) hangs indefinitely at
`tests/test_model.py::test_set_seed_makes_training_reproducible`, the first test that executes a
Keras model. It is a deadlock, not slowness: ~8s of CPU accumulated over 13 minutes of wall clock,
against a documented ~30s baseline for the whole suite.

Split out of `s-06-interactive-dashboard`, whose Phase 1 success criterion 1.1 ("full suite passes")
cannot be met until this is fixed. s-06's Phases 3–5 verify through `model.predict` and inherit the
same blocker. Nothing in s-06 caused this — the hang reproduces in bare Python with neither pytest
nor s-06's warning-filter changes present.

### Diagnosis so far

Sampled native stack of the wedged process:

```
tensorflow::ProcessFunctionLibraryRuntime::RunSync
  -> absl::Notification::WaitForNotification()
    -> absl::Mutex::Block
```

with every Eigen thread-pool worker parked in `WaitForWork`.

scikit-learn ships a **vendored OpenMP runtime** at
`.venv/lib/python3.13/site-packages/sklearn/.dylibs/libomp.dylib` (reporting 10 threads via
`threadpoolctl`). Loading it before TensorFlow is what wedges Keras execution.

Environment: TensorFlow 2.21.0, Keras 3.15.0, Python 3.13.13, Darwin 25.5.0, 10 cores, CPU-only.

### Evidence (standalone scripts, no pytest)

| Setup | Result |
|---|---|
| `import keras`, then sklearn, then predict + fit | works — 0.05s / 0.27s |
| sklearn first, then keras, then predict | deadlock |
| sklearn first + `OMP_NUM_THREADS=1` | deadlock |
| sklearn first + `KMP_DUPLICATE_LIB_OK=TRUE` | deadlock |
| raw TF eager ops + `tf.function`, no sklearn | fine — 0.01–0.02s |
| Bi-GRU `fit`, `keras.utils.set_random_seed`, keras-first | fine — model architecture is innocent |

### Root cause

**It is not sklearn specifically — it is numpy/pandas too.** The rule is: *Keras must be imported
before numpy and pandas*, not merely before scikit-learn.

That is why every `pytest_configure`-based fix failed. An import tracer installed at plugin-import
time showed the preload genuinely winning the race against sklearn (`keras` #1, `tensorflow` #2,
`sklearn` #3) and the suite deadlocking anyway. `tests/conftest.py` imports numpy and pandas at
module scope, and initial conftests load *before* `pytest_configure` — so no hook-based preload can
ever be early enough.

Confirmed standalone, no pytest involved:

- keras -> full mlpp stack (numpy, pandas, sklearn, joblib) -> exact test body: **passes**, fits in 2.43s / 2.30s
- numpy, pandas -> keras -> sklearn -> same test body: **deadlocks**

Ruled out as causes or cures: pytest output capture (`-s`), assertion rewriting (`--assert=plain`),
the warnings plugin (`-p no:warnings`, so the `filterwarnings` config is innocent), `OMP_NUM_THREADS=1`,
`KMP_DUPLICATE_LIB_OK=TRUE`, `TF_NUM_INTEROP_THREADS`/`TF_NUM_INTRAOP_THREADS=1`, and pinning
TensorFlow's inter/intra-op pools to 1 via the API. Model architecture is innocent: Bi-GRU `fit`,
`keras.utils.set_random_seed`, and raw TF eager/`tf.function` ops all run fine on their own.

### It is a dependency defect, not an import-order rule

An import-order guard in `tests/conftest.py` was tried first and rejected. It fixed the suite up to
`test_notebook.py`, then hung there — nbclient runs the notebook in an `ipykernel_launcher`
subprocess that inherits no guard. That exposed the real scope: **the deadlock is not confined to
tests.** `predict_cli.py:16` imports pandas at module scope and Keras lazily inside `_score`, which
is exactly the fatal order, so `mlpp-predict` deadlocks too — and the Streamlit dashboard s-06 is
about to build would inherit it, since Streamlit imports pandas long before `mlpp.predict`.

A minimal mlpp-free reproduction (numpy/pandas -> keras -> sklearn -> `fit`) isolated it to a
**version pair**:

| tensorflow | numpy | result |
|---|---|---|
| 2.21.0 | 2.5.1 | deadlock |
| 2.20.0 | 2.5.1 | ok, 0.34s |
| 2.21.0 | 2.3.5 | ok, 0.33s |

Either half avoids it. TensorFlow is the one pinned, because pinning numpy below 2.4 would drag
pandas back to 2.x and the project documents pandas 3.x.

### The fix

`apps/backend/pyproject.toml`: `tensorflow>=2.20,<2.22` -> `tensorflow>=2.20,<2.21`, with a comment
recording that the upper bound is a defect exclusion rather than conservatism, and that it should be
revisited when TF 2.22 ships. Re-locked and synced: `tensorflow 2.21.0 -> 2.20.0`.

No production code changes, no import-order rule for future entrypoints to remember, and the
TensorFlow-free module split is untouched.

### Verification

- Full suite: **145 passed, exit 0**, in 272s. Previously an unbounded hang at test 26.
- Fast suite: 98 passed / 47 deselected in 3.53s, still TensorFlow-free.
- `uv lock --check` clean, `mypy src/mlpp` clean, `ruff check` + `ruff format --check` clean.

Note for the record: the full suite takes **4m32s**, not the ~30s `CLAUDE.md` claims. The figure was
stale or measured elsewhere; nothing here made it slower, since the previous state was an infinite
hang rather than a slow pass. Worth correcting in `CLAUDE.md` separately.

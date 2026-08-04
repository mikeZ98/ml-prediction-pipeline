---
change_id: s-06-interactive-dashboard
title: Interactive Streamlit dashboard for model introspection and inference
status: implemented
created: 2026-07-31
updated: 2026-08-04
---

## Notes

Delivers the four surfaces described in `context/foundation/prd.md`: session
introspection, global permutation feature importance, single-row inference and
batch CSV/Parquet inference. Three are wrappers over existing capability;
feature importance is new domain logic and is built as a TensorFlow-free
library module rather than inside the interface.

Upstream artifacts: `context/foundation/shape-notes.md`,
`context/foundation/prd.md`, `context/foundation/stack-assessment.md`,
`context/foundation/health-check.md`.

## Outcome

All five phases landed: `f91206d` (p1), `0b87e6d` (p2), `ae29a3a` (p3), `5fcc11d` (p4),
`a609721` (p5). Suite grew 145 -> 195 tests; the fast suite stayed TensorFlow-free
throughout (98 -> 141 tests, ~4s).

### Deviations from the plan, and why

- **A blocker was split out as `s-07-tf-sklearn-deadlock` (`d89e860`).** Phase 1's
  criterion 1.1 could not pass: the full suite hung forever on the first test that
  executed a Keras model. TensorFlow 2.21 paired with numpy 2.5 deadlocks once a
  threaded OpenMP/BLAS runtime loads before TensorFlow — which the TensorFlow-free
  module split guarantees. It also broke `mlpp-predict` and would have broken this
  dashboard. Fixed by pinning `tensorflow<2.21`.
- **Plan premise now stale: `OUTPUTS/` holds no incomplete sessions.** All eight
  directories have valid manifests, so FR-002's error surface was verified against
  synthetic broken sessions instead. Two are now permanent tests.
- **FR-022's range check lives in `preprocess.flag_out_of_range()`, not in the panel.**
  The plan placed it in `single.py`, but it computes, and CLAUDE.md scopes `dashboard/`
  to rendering. Same for the download frame: `predict_cli.prediction_frame()` was
  promoted from private and is shared, so a download and `mlpp-predict` cannot
  disagree on column layout.
- **`.streamlit/config.toml` added (not in the plan).** Streamlit binds every
  interface by default; on first launch it advertised an External URL on the machine's
  public address. FR-021 scopes the dashboard to the local machine, so the bind address
  is pinned to loopback.
- **5.7 is met approximately.** `st.bar_chart` cannot draw error bars, so `std_drop` is
  shown as a column beside `mean R² drop` rather than as whiskers. Real error bars
  would mean adding a plotting library for one chart.

### Follow-ups this change did not take

- `CLAUDE.md` still says "TensorFlow 2.20–2.21" and claims a ~30s full suite; both are
  now wrong (2.21 is excluded, and the suite takes ~4m). Left alone deliberately to
  avoid re-entangling s-06 and s-07 edits to the same file.
- The joblib/numpy deprecation tripwire armed in Phase 1 still carries its scoped
  ignore — the version decision remains its own change.

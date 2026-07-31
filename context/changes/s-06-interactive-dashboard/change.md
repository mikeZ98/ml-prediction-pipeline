---
change_id: s-06-interactive-dashboard
title: Interactive Streamlit dashboard for model introspection and inference
status: planned
created: 2026-07-31
updated: 2026-07-31
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

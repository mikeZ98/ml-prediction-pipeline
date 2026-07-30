---
change_id: s-05-notebook-consolidation
title: Notebook consolidation — make the demo notebook verified and presentable
status: implemented
created: 2026-07-30
updated: 2026-07-30
archived_at: null
---

## Notes

Repository polish for recruiter-facing visibility, framed as notebook
consolidation: audit and reduce to one production-grade notebook, extract
remaining logic into typed tested modules, keep every gate green.

The audit found one notebook with no extractable logic — but also found it
broken on `main` (s-04 moved `DataConfig`'s column fields into `ColumnConfig`
and the notebook still uses the old signature). Immediately followed by s-06,
a lightweight Streamlit/Taipy dashboard in `apps/ui` consuming the same
`LoadedSession` scoring core and `OUTPUTS/` manifests.

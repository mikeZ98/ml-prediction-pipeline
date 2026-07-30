---
change_id: s-04-production-inference-seam
title: Select the next epic — inference seam vs. new architectures vs. dashboard
status: archived
created: 2026-07-30
updated: 2026-07-30
archived_at: 2026-07-30T10:05:13Z
---

## Notes

Framing step for the post-s-03 epic decision. Three candidates were tabled:
(A) production inference seam (`mlpp-predict` CLI + FastAPI), (B) next-gen ML
baselines (LightGBM/CatBoost, SSM, zero-shot TimesFM/Chronos), (C) interactive
business dashboard. Target use-case: energy/OZE load forecasting and industrial
IoT predictive maintenance. No external consumer is waiting; all three are
expected to happen, so the real question is ordering.

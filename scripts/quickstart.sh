#!/usr/bin/env bash
# One-command setup + training run. uv only — never pip/venv/conda.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "==> syncing apps/backend"
uv sync --project apps/backend --all-groups

echo "==> training (pass extra flags through, e.g. $0 --epochs 20)"
uv run --project apps/backend mlpp-train "$@"

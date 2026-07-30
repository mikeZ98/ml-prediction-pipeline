#!/usr/bin/env python3
"""Verify the committed notebook's *source* still matches scripts/build_notebook.py.

Compares cell type, id and source plus notebook-level metadata, and deliberately
ignores `outputs` and `execution_count`. That distinction is the point: the
committed notebook carries executed outputs so a reader sees results without
running anything, but its code must still come from the generator.

This replaces a whole-file `git diff`, which could not tell a hand-edited cell
apart from a legitimately executed one — and which passed green while the
notebook was broken, because it only ever checked formatting fidelity.

    uv run --no-project --python 3.12 python scripts/check_notebook_drift.py

Exits 0 when the source matches, 1 otherwise. Stdlib only, so it runs without
syncing the project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_notebook import NOTEBOOK, REPO_ROOT, build

REGENERATE_HINT = "uv run --project apps/backend python scripts/build_notebook.py"

#: Per-cell keys that must match. `outputs` and `execution_count` are excluded by
#: design — they are products of execution, not of the generator.
COMPARED_CELL_KEYS = ("cell_type", "id", "source")


def _cell_fingerprint(cell: dict[str, Any]) -> dict[str, Any]:
    return {key: cell.get(key) for key in COMPARED_CELL_KEYS}


def _differences(expected: dict[str, Any], committed: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    exp_cells = expected["cells"]
    got_cells = committed.get("cells", [])
    if len(exp_cells) != len(got_cells):
        problems.append(
            f"cell count: generator produces {len(exp_cells)}, committed has {len(got_cells)}"
        )

    for index, (exp, got) in enumerate(zip(exp_cells, got_cells)):
        exp_fp, got_fp = _cell_fingerprint(exp), _cell_fingerprint(got)
        if exp_fp != got_fp:
            for key in COMPARED_CELL_KEYS:
                if exp_fp[key] != got_fp[key]:
                    problems.append(f"cell {index}: {key} differs from the generator")

    for key in ("metadata", "nbformat", "nbformat_minor"):
        if expected[key] != committed.get(key):
            problems.append(f"notebook {key} differs from the generator")

    return problems


def main() -> int:
    if not NOTEBOOK.is_file():
        print(
            f"error: {NOTEBOOK.relative_to(REPO_ROOT)} does not exist", file=sys.stderr
        )
        print(f"Generate it with: {REGENERATE_HINT}", file=sys.stderr)
        return 1

    committed = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    problems = _differences(build(), committed)

    if problems:
        rel = NOTEBOOK.relative_to(REPO_ROOT)
        print(
            f"error: {rel} is out of sync with scripts/build_notebook.py",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nThe notebook is generated and must never be hand-edited.",
            file=sys.stderr,
        )
        print(f"Regenerate it with: {REGENERATE_HINT}", file=sys.stderr)
        print(
            "(then re-execute it if you need the committed outputs refreshed)",
            file=sys.stderr,
        )
        return 1

    cells = len(committed["cells"])
    with_outputs = sum(1 for c in committed["cells"] if c.get("outputs"))
    print(
        f"notebook source is in sync ({cells} cells, {with_outputs} carrying outputs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

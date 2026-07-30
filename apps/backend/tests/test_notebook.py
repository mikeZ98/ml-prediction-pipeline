"""Executes the committed notebook — the gate whose absence let it rot.

The notebook is generated from string literals in `scripts/build_notebook.py`, so
`mypy` and `ruff` see strings rather than code, and the CI drift check only proves
the source matches its generator. Nothing executed it, and two defects reached
`main` as a result: a config cell calling a signature that no longer existed, and
an IFrame path that crashed when the notebook ran from its own directory.

These tests are deliberately written *without* a skip-if-nbclient-missing guard.
A silent skip would restore exactly the false confidence this module exists to
remove — CI syncs the `notebook` dependency group so the import must succeed.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = REPO_ROOT / "notebooks" / "01_train.ipynb"

#: The committed notebook names the `mlpp` kernel, which only exists once a human
#: runs `ipykernel install --name mlpp`. Executing here overrides it rather than
#: depending on that side effect.
KERNEL = "python3"

#: Generous: the notebook trains a real model. Measured at ~18s locally.
TIMEOUT_SECONDS = 900


def _execute(working_dir: Path) -> nbformat.NotebookNode:
    """Run every cell with the kernel's cwd set to `working_dir`.

    Both arguments matter and neither can be left to the default: the kernel name
    must be overridden (the committed one is unregistered) and the working
    directory decides whether the notebook's own path handling holds up.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(
        notebook,
        timeout=TIMEOUT_SECONDS,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(working_dir)}},
    ).execute()
    return notebook


def test_notebook_exists() -> None:
    assert NOTEBOOK.is_file(), f"expected a committed notebook at {NOTEBOOK}"


def test_notebook_executes_from_repo_root() -> None:
    """Every cell runs without raising. This is the whole point of the module."""
    executed = _execute(REPO_ROOT)
    assert executed.cells, "notebook has no cells"


def test_notebook_executes_from_its_own_directory() -> None:
    """Regression pin: JupyterLab's cwd is the notebook's directory, not the repo root.

    The IFrame cell used to resolve its path against `Path.cwd()`, which raised
    `ValueError: ... is not in the subpath of .../notebooks` under exactly this
    condition — the normal way a human opens the notebook.
    """
    executed = _execute(NOTEBOOK.parent)
    assert executed.cells, "notebook has no cells"


def test_every_code_cell_carries_an_id() -> None:
    """nbformat warns that a missing cell id will become a hard error."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    missing = [i for i, cell in enumerate(notebook.cells) if not cell.get("id")]
    assert not missing, f"cells without an id: {missing}"

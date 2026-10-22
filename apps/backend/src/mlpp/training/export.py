"""PyTorch -> ONNX export, verified against ONNX Runtime before it is trusted.

The exported graph is what every scorer runs (`mlpp.inference.onnx_model`), so an
export that silently diverged from the trained network would poison serving, the
predict CLI and the dashboard at once. `export_onnx` therefore re-scores a sample
through ONNX Runtime and refuses the file when the two disagree.

Uses the `torch.export`-based exporter (the default since torch 2.9) with a dynamic
batch axis; the feature axis stays fixed, which is what lets `load_onnx_model`
cross-check a graph against its session's manifest.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
import torch

from mlpp.errors import ArtifactError
from mlpp.inference.onnx_model import METADATA_N_PARAMETERS, METADATA_OPSET, OnnxModel
from mlpp.training.loop import predict_scaled
from mlpp.training.model import CnnBiGruRegressor, count_parameters

ONNX_INPUT: Final = "features"
ONNX_OUTPUT: Final = "prediction"
ONNX_OPSET: Final = 18
#: float32 kernels differ between eager PyTorch and ONNX Runtime by ~1e-7; this is
#: comfortably above that noise and far below any real divergence.
PARITY_ATOL: Final = 1e-4


def export_onnx(
    model: CnnBiGruRegressor,
    path: Path,
    *,
    sample: np.ndarray | None = None,
    atol: float = PARITY_ATOL,
) -> Path:
    """Export `model` to `path` and verify it on `sample` (or a random batch).

    The model is copied to the CPU for export, so a GPU-resident model is left
    untouched. Raises ArtifactError when ONNX Runtime's output differs from PyTorch's
    by more than `atol` on any row.
    """
    import onnx

    cpu_model = copy.deepcopy(model).cpu().eval()
    n_features = cpu_model.n_features
    # Batch 2, not 1: torch.export specialises a size-1 dimension to a constant.
    example = torch.zeros(2, n_features, 1)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _quiet_exporter():
        torch.onnx.export(
            cpu_model,
            (example,),
            str(path),
            input_names=[ONNX_INPUT],
            output_names=[ONNX_OUTPUT],
            opset_version=ONNX_OPSET,
            dynamo=True,
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
            external_data=False,
            optimize=True,
            verbose=False,
        )

    graph = onnx.load(str(path))
    onnx.helper.set_model_props(
        graph,
        {
            METADATA_N_PARAMETERS: str(count_parameters(cpu_model)),
            METADATA_OPSET: str(ONNX_OPSET),
        },
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))

    probe = sample if sample is not None else _random_batch(n_features)
    verify_parity(cpu_model, path, probe, atol=atol)
    return path


def verify_parity(
    model: CnnBiGruRegressor, path: Path, sample: np.ndarray, *, atol: float
) -> float:
    """Max |PyTorch - ONNX Runtime| over `sample`; raises ArtifactError above `atol`."""
    expected = predict_scaled(model, sample)
    actual = OnnxModel.from_path(path).predict(sample)
    worst = float(np.max(np.abs(expected - actual))) if len(sample) else 0.0
    if not worst <= atol:  # also catches NaN
        raise ArtifactError(
            f"{path}: ONNX output differs from PyTorch by {worst:.3g} (tolerance {atol:.1g}); "
            "the exported graph is not a faithful copy of the trained model"
        )
    return worst


def _random_batch(n_features: int, rows: int = 64) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((rows, n_features, 1)).astype(np.float32)


#: Loggers the exporter stack writes INFO chatter to (graph-pass notices, missing
#: torchvision ops). Raised to ERROR for the duration of an export only.
_EXPORTER_LOGGERS: Final = ("torch.onnx", "torch.export", "onnx_ir", "onnxscript")


@contextlib.contextmanager
def _quiet_exporter() -> Iterator[None]:
    """Silence the exporter's own chatter (pass logs, deprecations) during one export."""
    loggers = [logging.getLogger(name) for name in _EXPORTER_LOGGERS]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for logger, level in zip(loggers, previous, strict=True):
            logger.setLevel(level)

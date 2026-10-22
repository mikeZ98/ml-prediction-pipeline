"""ONNX Runtime wrapper: the one place a trained model is executed for scoring.

Training exports the network to ONNX (`mlpp.training.export`); everything that
scores — the API, `mlpp-predict`, the dashboard, permutation importance — runs that
graph here. Serving therefore needs `onnxruntime` and nothing from the training
stack, which keeps the container small and the import cheap.

`OnnxModel` checks its input before running anything. ONNX Runtime does reject a
mis-shaped tensor, but with a C++ message naming graph internals; the checks here
turn the same failure into a `SchemaError` a caller can act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from mlpp.errors import ArtifactError, InferenceError, SchemaError
from mlpp.session import ROLE_BEST_MODEL, LoadedSession

#: Custom metadata keys written into the graph at export time.
METADATA_N_PARAMETERS: Final = "mlpp.n_parameters"
METADATA_OPSET: Final = "mlpp.opset"

_FLOAT_TENSOR: Final = "tensor(float)"
_CPU: Final = ("CPUExecutionProvider",)

Dim = int | str | None


@dataclass(frozen=True, slots=True)
class ModelDescription:
    """What the graph declares about itself — for introspection, never for scoring."""

    input_name: str
    input_shape: tuple[Dim, ...]
    output_name: str
    output_shape: tuple[Dim, ...]
    producer: str
    opset: int | None
    n_parameters: int | None
    source: Path | None


class OnnxModel:
    """A loaded ONNX graph mapping X `(rows, n_features, 1)` to one value per row.

    `InferenceSession.run` is thread-safe, so one instance serves concurrent
    requests; the API runs it in worker threads to keep the event loop free.
    """

    def __init__(self, session: Any, *, source: Path | None = None) -> None:
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ArtifactError(
                f"{source or 'ONNX graph'}: expected exactly one input and one output, "
                f"got {len(inputs)} and {len(outputs)}"
            )
        node = inputs[0]
        shape = tuple(node.shape)
        if node.type != _FLOAT_TENSOR:
            raise ArtifactError(f"{source or 'ONNX graph'}: input must be float32, got {node.type}")
        # A symbolic (str/None) trailing dim is tolerated; a fixed one must be 1.
        if len(shape) != 3 or (isinstance(shape[2], int) and shape[2] != 1):
            raise ArtifactError(
                f"{source or 'ONNX graph'}: input must be shaped (batch, n_features, 1), "
                f"got {list(shape)}"
            )

        self._session = session
        self._source = source
        self._input_name: str = node.name
        self._input_shape: tuple[Dim, ...] = shape
        self._output_name: str = outputs[0].name
        self._output_shape: tuple[Dim, ...] = tuple(outputs[0].shape)
        self._n_features: int | None = shape[1] if isinstance(shape[1], int) else None

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        intra_op_threads: int | None = None,
        inter_op_threads: int | None = None,
    ) -> OnnxModel:
        """Open `path` on the CPU provider, translating runtime failures to ArtifactError.

        Thread counts default to ONNX Runtime's own choice (one per physical core).
        Pin `intra_op_threads` when several workers share a host, or they oversubscribe.
        """
        import onnxruntime as ort

        if not path.is_file():
            raise ArtifactError(f"ONNX model not found: {path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3  # errors only; warnings go to stderr unbidden
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
        if inter_op_threads is not None:
            options.inter_op_num_threads = inter_op_threads
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=list(_CPU))
        except Exception as exc:  # InvalidProtobuf, NoSuchFile, Fail: bare pybind11 types
            raise ArtifactError(f"cannot load ONNX model {path}: {exc}") from exc
        return cls(session, source=path)

    @property
    def n_features(self) -> int | None:
        """Feature-axis length fixed in the graph, or None when it is symbolic."""
        return self._n_features

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_name(self) -> str:
        return self._output_name

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Score `x` of shape `(rows, n_features, 1)`; returns a float32 array `(rows,)`.

        Values come back in whatever units the graph was trained on (scaled, for
        mlpp sessions) — inverting the target scaling is the preprocessor's job.

        Raises SchemaError for an input of the wrong shape and InferenceError when the
        runtime fails on a well-formed one.
        """
        array = np.asarray(x)
        if array.ndim != 3 or array.shape[2] != 1:
            raise SchemaError(
                f"model input must be shaped (rows, n_features, 1), got {array.shape}"
            )
        if self._n_features is not None and array.shape[1] != self._n_features:
            raise SchemaError(
                f"model expects {self._n_features} features, input has {array.shape[1]}"
            )
        rows = array.shape[0]
        if rows == 0:
            return np.empty((0,), dtype=np.float32)

        batch = np.ascontiguousarray(array, dtype=np.float32)
        try:
            (raw,) = self._session.run([self._output_name], {self._input_name: batch})
        except Exception as exc:  # the runtime raises bare pybind11 error types
            raise InferenceError(f"ONNX Runtime failed to score {rows} row(s): {exc}") from exc

        out = np.asarray(raw, dtype=np.float32)
        if out.shape[0] != rows or out.size != rows:
            raise InferenceError(
                f"model returned shape {out.shape} for {rows} row(s); expected one value per row"
            )
        return out.reshape(rows)

    def describe(self) -> ModelDescription:
        meta = self._session.get_modelmeta()
        custom: dict[str, str] = dict(getattr(meta, "custom_metadata_map", {}) or {})
        return ModelDescription(
            input_name=self._input_name,
            input_shape=self._input_shape,
            output_name=self._output_name,
            output_shape=self._output_shape,
            producer=str(getattr(meta, "producer_name", "") or "unknown"),
            opset=_int_or_none(custom.get(METADATA_OPSET)),
            n_parameters=_int_or_none(custom.get(METADATA_N_PARAMETERS)),
            source=self._source,
        )


def load_onnx_model(
    session: LoadedSession,
    *,
    intra_op_threads: int | None = None,
    inter_op_threads: int | None = None,
) -> OnnxModel:
    """Load the session's best model, resolving its filename through the manifest.

    The filename is never spelled here — `session.py` owns every name in a session
    directory, so this asks the manifest inventory for the `best_model` role. The
    graph's fixed feature count is cross-checked against the manifest's feature
    names, so a graph and contract from different runs cannot be paired silently.
    """
    filenames = session.manifest.filenames_for(ROLE_BEST_MODEL)
    if not filenames:
        raise ArtifactError(
            f"{session.session_dir}: manifest records no {ROLE_BEST_MODEL!r} artifact; "
            "the session has no model to score with"
        )
    if len(filenames) > 1:
        raise ArtifactError(
            f"{session.session_dir}: manifest records {len(filenames)} {ROLE_BEST_MODEL!r} "
            f"artifacts ({list(filenames)}); expected exactly one"
        )

    model = OnnxModel.from_path(
        session.session_dir / filenames[0],
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )
    expected = len(session.manifest.features.feature_names)
    if model.n_features is not None and model.n_features != expected:
        raise ArtifactError(
            f"{session.session_dir}: the ONNX graph takes {model.n_features} features but the "
            f"manifest lists {expected}; the model and contract come from different runs"
        )
    return model


def _int_or_none(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None

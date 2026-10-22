"""ONNX Runtime wrapper: mocked sessions for the contract, a tiny real graph for the runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import make_mock_ort_session, ort_node

from mlpp.config import ColumnConfig
from mlpp.errors import ArtifactError, InferenceError, SchemaError
from mlpp.inference.onnx_model import OnnxModel, load_onnx_model
from mlpp.preprocess import Preprocessor
from mlpp.session import (
    BEST_MODEL_FILE,
    ROLE_BEST_MODEL,
    FeatureContract,
    LoadedSession,
    SessionWriter,
    load_session,
    write_fitted_state,
)


def test_predict_binds_the_graph_input_and_returns_one_value_per_row() -> None:
    session = make_mock_ort_session()
    model = OnnxModel(session)
    x = np.arange(12, dtype=np.float64).reshape(4, 3, 1)

    out = model.predict(x)

    np.testing.assert_allclose(out, x.sum(axis=1).ravel())
    assert out.shape == (4,)
    assert out.dtype == np.float32
    (names, feeds), _ = session.run.call_args
    assert names == ["prediction"]
    assert feeds["features"].dtype == np.float32, "inputs must be cast to the graph's dtype"
    assert feeds["features"].flags["C_CONTIGUOUS"]


def test_a_non_contiguous_view_is_copied_before_scoring() -> None:
    session = make_mock_ort_session()
    x = np.asfortranarray(np.ones((5, 3, 1), dtype=np.float32))
    OnnxModel(session).predict(x)
    (_, feeds), _ = session.run.call_args
    assert feeds["features"].flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("shape", [(4, 3), (4, 3, 2), (4, 2, 1), (4, 4, 1)])
def test_a_mis_shaped_input_is_a_schema_error_before_the_runtime_runs(
    shape: tuple[int, ...],
) -> None:
    session = make_mock_ort_session()
    with pytest.raises(SchemaError):
        OnnxModel(session).predict(np.zeros(shape, dtype=np.float32))
    session.run.assert_not_called()


def test_a_symbolic_feature_axis_accepts_any_width() -> None:
    model = OnnxModel(make_mock_ort_session(n_features="n"))
    assert model.n_features is None
    assert model.predict(np.ones((2, 7, 1), dtype=np.float32)).tolist() == [7.0, 7.0]


def test_an_empty_batch_short_circuits() -> None:
    session = make_mock_ort_session()
    out = OnnxModel(session).predict(np.zeros((0, 3, 1), dtype=np.float32))
    assert out.shape == (0,)
    session.run.assert_not_called()


def test_a_runtime_failure_becomes_an_inference_error() -> None:
    def explode(names: Any, feeds: Any) -> Any:
        raise RuntimeError("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT")

    with pytest.raises(InferenceError, match="failed to score 2 row") as excinfo:
        OnnxModel(make_mock_ort_session(run=explode)).predict(np.zeros((2, 3, 1)))
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_an_output_that_is_not_one_value_per_row_is_rejected() -> None:
    session = make_mock_ort_session(run=lambda names, feeds: [np.zeros((3, 2), dtype=np.float32)])
    with pytest.raises(InferenceError, match="one value per row"):
        OnnxModel(session).predict(np.zeros((3, 3, 1)))


@pytest.mark.parametrize(
    "inputs",
    [
        [ort_node("a", ["batch", 3, 1]), ort_node("b", ["batch", 3, 1])],
        [ort_node("features", ["batch", 3, 1], "tensor(double)")],
        [ort_node("features", ["batch", 3])],
        [ort_node("features", ["batch", 3, 2])],
    ],
)
def test_a_graph_with_the_wrong_io_contract_is_rejected_at_load(
    inputs: list[SimpleNamespace],
) -> None:
    with pytest.raises(ArtifactError):
        OnnxModel(make_mock_ort_session(inputs=inputs))


def test_describe_reads_the_export_metadata() -> None:
    description = OnnxModel(make_mock_ort_session(), source=Path("m.onnx")).describe()
    assert description.input_name == "features"
    assert description.input_shape == ("batch", 3, 1)
    assert description.producer == "pytorch"
    assert description.opset == 18
    assert description.n_parameters == 1234
    assert description.source == Path("m.onnx")


def test_describe_tolerates_a_graph_without_metadata() -> None:
    session = make_mock_ort_session()
    session.get_modelmeta.return_value = SimpleNamespace(producer_name="", custom_metadata_map={})
    description = OnnxModel(session).describe()
    assert (description.opset, description.n_parameters, description.producer) == (
        None,
        None,
        "unknown",
    )


# --- real ONNX Runtime ------------------------------------------------------------------


def test_a_real_graph_scores_through_onnx_runtime(
    tmp_path: Path, linear_onnx: Callable[..., Path]
) -> None:
    path = linear_onnx(tmp_path / "linear.onnx", [1.0, 2.0, -1.0], bias=0.5)
    model = OnnxModel.from_path(path, intra_op_threads=1, inter_op_threads=1)
    x = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 4.0]], dtype=np.float32)[..., None]

    np.testing.assert_allclose(model.predict(x), [2.5, 0.5], rtol=1e-6)
    assert model.n_features == 3


def test_a_missing_file_is_an_artifact_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        OnnxModel.from_path(tmp_path / "absent.onnx")


def test_a_corrupt_file_is_an_artifact_error_not_a_runtime_crash(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"definitely not a protobuf")
    with pytest.raises(ArtifactError, match="cannot load ONNX model"):
        OnnxModel.from_path(corrupt)


# --- manifest resolution ----------------------------------------------------------------


def _session_with_model(
    tmp_path: Path,
    frame: pd.DataFrame,
    column_cfg: ColumnConfig,
    linear_onnx: Callable[..., Path],
    *,
    n_weights: int | None = None,
    model_files: tuple[str, ...] = (BEST_MODEL_FILE,),
) -> LoadedSession:
    pre = Preprocessor(column_cfg).fit(frame)
    writer = SessionWriter(tmp_path / "session")
    writer.set_features(
        FeatureContract(
            pre.schema.numeric, pre.schema.categorical, pre.schema.output, pre.feature_names
        )
    )
    write_fitted_state(writer, pre.fitted_state)
    for filename in model_files:
        linear_onnx(
            writer.register(ROLE_BEST_MODEL, filename), [1.0] * (n_weights or pre.n_features)
        )
    writer.flush()
    return load_session(writer.session_dir, column_cfg)


def test_load_onnx_model_resolves_the_file_through_the_manifest(
    tmp_path: Path, frame: pd.DataFrame, column_cfg: ColumnConfig, linear_onnx: Callable[..., Path]
) -> None:
    session = _session_with_model(tmp_path, frame, column_cfg, linear_onnx)
    model = load_onnx_model(session)
    x, _ = session.preprocessor.transform(frame)
    assert model.predict(x).shape == (len(frame),)


def test_a_session_without_a_model_role_fails_loudly(
    tmp_path: Path, frame: pd.DataFrame, column_cfg: ColumnConfig, linear_onnx: Callable[..., Path]
) -> None:
    session = _session_with_model(tmp_path, frame, column_cfg, linear_onnx, model_files=())
    with pytest.raises(ArtifactError, match="no 'best_model' artifact"):
        load_onnx_model(session)


def test_an_ambiguous_model_role_fails_loudly(
    tmp_path: Path, frame: pd.DataFrame, column_cfg: ColumnConfig, linear_onnx: Callable[..., Path]
) -> None:
    session = _session_with_model(
        tmp_path, frame, column_cfg, linear_onnx, model_files=("a.onnx", "b.onnx")
    )
    with pytest.raises(ArtifactError, match="expected exactly one"):
        load_onnx_model(session)


def test_a_graph_from_a_different_run_is_rejected(
    tmp_path: Path, frame: pd.DataFrame, column_cfg: ColumnConfig, linear_onnx: Callable[..., Path]
) -> None:
    session = _session_with_model(tmp_path, frame, column_cfg, linear_onnx, n_weights=7)
    with pytest.raises(ArtifactError, match="different runs"):
        load_onnx_model(session)

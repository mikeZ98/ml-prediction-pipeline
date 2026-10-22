"""HTTP contract of the inference API.

The ONNX session is mocked (`conftest.make_mock_ort_session`): every test runs the
real FastAPI app, real request validation, the real fitted preprocessor and the real
inverse target transform, with only the graph execution replaced by a deterministic
stand-in. That keeps the suite fast and lets each test assert exact predictions. One
lifespan test at the end loads a genuine (tiny) ONNX file through ONNX Runtime.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx2 as httpx
import numpy as np
import pandas as pd
import pytest
from conftest import make_mock_ort_session
from fastapi.testclient import TestClient

from mlpp.config import ColumnConfig
from mlpp.errors import ArtifactError
from mlpp.inference.api import REQUEST_ID_HEADER, create_app
from mlpp.inference.onnx_model import OnnxModel
from mlpp.inference.service import InferenceService
from mlpp.inference.settings import ServiceSettings
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

NUMERIC = ("feature_01", "feature_02", "comp_active")
CATEGORICAL = ("site",)
MAX_ROWS = 5
MAX_BODY = 4096


# --- fixtures ---------------------------------------------------------------------------


@pytest.fixture
def api_columns() -> ColumnConfig:
    return ColumnConfig(
        input_columns=NUMERIC + CATEGORICAL,
        output_column="target",
        categorical_columns=CATEGORICAL,
    )


@pytest.fixture
def session(
    tmp_path: Path,
    rng: np.random.Generator,
    api_columns: ColumnConfig,
    linear_onnx: Callable[..., Path],
) -> LoadedSession:
    """A real session on disk: fitted scalers, a one-hot `site`, and a real ONNX file."""
    n = 64
    frame = pd.DataFrame(
        {
            "feature_01": rng.normal(size=n),
            "feature_02": rng.normal(5.0, 2.0, size=n),
            "comp_active": rng.integers(0, 2, size=n),
            "site": rng.choice(["north", "south"], size=n),
            "target": rng.normal(100.0, 10.0, size=n),
        }
    )
    pre = Preprocessor(api_columns).fit(frame)
    writer = SessionWriter(tmp_path / "2026-01-01_00-00-00")
    writer.set_features(
        FeatureContract(
            pre.schema.numeric, pre.schema.categorical, pre.schema.output, pre.feature_names
        )
    )
    write_fitted_state(writer, pre.fitted_state)
    linear_onnx(writer.register(ROLE_BEST_MODEL, BEST_MODEL_FILE), [1.0] * pre.n_features)
    writer.flush()
    return load_session(writer.session_dir, api_columns)


@pytest.fixture
def ort_session(session: LoadedSession) -> MagicMock:
    return make_mock_ort_session(session.preprocessor.n_features)


@pytest.fixture
def service(session: LoadedSession, ort_session: MagicMock) -> InferenceService:
    return InferenceService(session, OnnxModel(ort_session), max_concurrency=2)


@pytest.fixture
def settings(session: LoadedSession) -> ServiceSettings:
    return ServiceSettings(
        session_dir=session.session_dir, max_batch_rows=MAX_ROWS, max_body_bytes=MAX_BODY
    )


@pytest.fixture
def client(settings: ServiceSettings, service: InferenceService) -> Iterator[TestClient]:
    app = create_app(settings, service=service)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"feature_01": 0.5, "feature_02": 4.0, "comp_active": 1, "site": "north"}
    row.update(overrides)
    return row


def _post(client: TestClient, body: Any, **kwargs: Any) -> httpx.Response:
    return client.post("/v1/predict", json=body, **kwargs)


def _expected(session: LoadedSession, records: list[dict[str, Any]]) -> np.ndarray:
    """What the mocked graph (row sum of X) yields after the real inverse transform."""
    x, _ = session.preprocessor.transform(pd.DataFrame(records))
    return session.preprocessor.inverse_target(x.sum(axis=1))


def _assert_error(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body: dict[str, Any] = response.json()
    assert body["error"] == code
    assert set(body) == {"error", "message", "request_id", "details"}
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert "Traceback" not in response.text
    return body


# --- health and metadata ----------------------------------------------------------------


def test_liveness_never_depends_on_the_model(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_the_served_session(client: TestClient, session: LoadedSession) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["model"] == {
        "session": session.session_dir.name,
        "created": session.manifest.created,
        "schema_version": session.manifest.schema_version,
    }


def test_before_startup_the_service_is_unavailable_but_alive(
    settings: ServiceSettings, service: InferenceService
) -> None:
    """Without lifespan (no `with`), the model is not loaded: ready and predict say 503."""
    unstarted = TestClient(create_app(settings, service=service), raise_server_exceptions=False)
    assert unstarted.get("/health/live").status_code == 200
    _assert_error(unstarted.get("/health/ready"), 503, "model_unavailable")
    _assert_error(_post(unstarted, {"records": [_row()]}), 503, "model_unavailable")


def test_model_info_publishes_the_contract(client: TestClient, session: LoadedSession) -> None:
    info = client.get("/v1/model").json()
    assert info["numeric_columns"] == list(NUMERIC)
    assert info["categorical_columns"] == list(CATEGORICAL)
    assert info["feature_names"] == list(session.manifest.features.feature_names)
    assert info["max_batch_rows"] == MAX_ROWS
    assert info["runtime"] == {
        "input_name": "features",
        "output_name": "prediction",
        "opset": 18,
        "producer": "pytorch",
        "n_parameters": 1234,
    }


# --- the happy path ---------------------------------------------------------------------


def test_predict_returns_inverse_transformed_values_in_request_order(
    client: TestClient, session: LoadedSession
) -> None:
    records = [_row(), _row(feature_01=-2.0, site="south"), _row(comp_active=0)]
    response = _post(client, {"records": records})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_records"] == 3
    np.testing.assert_allclose(body["predictions"], _expected(session, records), rtol=1e-6)
    assert body["unseen_levels"] == {}
    assert body["warnings"] == []
    # Predictions live in the target's units (~100), not the scaled space (~0).
    assert all(50 < p < 150 for p in body["predictions"])


def test_the_mocked_graph_receives_a_float32_tensor_of_the_contract_shape(
    client: TestClient, ort_session: MagicMock, session: LoadedSession
) -> None:
    _post(client, {"records": [_row(), _row()]})
    (names, feeds), _ = ort_session.run.call_args
    assert names == ["prediction"]
    tensor = feeds["features"]
    assert tensor.dtype == np.float32
    assert tensor.shape == (2, session.preprocessor.n_features, 1)


def test_a_full_batch_is_accepted(client: TestClient) -> None:
    response = _post(client, {"records": [_row()] * MAX_ROWS})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == MAX_ROWS


def test_an_unseen_categorical_level_scores_but_is_reported(client: TestClient) -> None:
    response = _post(client, {"records": [_row(site="east"), _row(), _row(site=3)]})
    assert response.status_code == 200
    body = response.json()
    assert body["unseen_levels"] == {"site": ["east", 3]}
    assert "2 row(s)" in body["warnings"][0]


# --- request IDs ------------------------------------------------------------------------


def test_a_request_id_is_generated_and_echoed(client: TestClient) -> None:
    response = _post(client, {"records": [_row()]})
    rid = response.headers[REQUEST_ID_HEADER]
    assert len(rid) == 32
    assert response.json()["request_id"] == rid


def test_a_well_formed_caller_request_id_is_kept(client: TestClient) -> None:
    response = _post(client, {"records": [_row()]}, headers={REQUEST_ID_HEADER: "trace-42.a:b"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-42.a:b"
    assert response.json()["request_id"] == "trace-42.a:b"


@pytest.mark.parametrize("unsafe", ["has space", "x" * 129, "semi;colon", "path/like", ""])
def test_an_unsafe_caller_request_id_is_replaced(client: TestClient, unsafe: str) -> None:
    response = client.get("/health/live", headers={REQUEST_ID_HEADER: unsafe})
    assert response.headers[REQUEST_ID_HEADER] != unsafe
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


# --- validation edge cases --------------------------------------------------------------


def _raw_row(drop: tuple[str, ...] = (), **raw: str) -> str:
    """One record as raw JSON text, so a test can send what `json.dumps` cannot (NaN)."""
    fields = {"feature_01": "1", "feature_02": "1", "comp_active": "1", "site": '"n"'}
    for name in drop:
        del fields[name]
    fields.update(raw)
    return "{" + ", ".join(f'"{k}": {v}' for k, v in fields.items()) + "}"


def _records(*rows: str) -> str:
    return '{"records": [' + ", ".join(rows) + "]}"


@pytest.mark.parametrize(
    ("raw_body", "loc", "error_type"),
    [
        (_records(_raw_row(drop=("comp_active",))), ["records", 0, "comp_active"], "missing"),
        (_records(_raw_row(x="1")), ["records", 0, "x"], "extra_forbidden"),
        (_records(_raw_row(target="9")), ["records", 0, "target"], "extra_forbidden"),
        (_records(_raw_row(feature_01="NaN")), ["records", 0, "feature_01"], "finite_number"),
        (_records(_raw_row(feature_01="Infinity")), ["records", 0, "feature_01"], "finite_number"),
        (_records(_raw_row(feature_01="1e999")), ["records", 0, "feature_01"], "finite_number"),
        (_records(_raw_row(feature_01='"1.5"')), ["records", 0, "feature_01"], "float_type"),
        (_records(_raw_row(comp_active="true")), ["records", 0, "comp_active"], "float_type"),
        (_records(_raw_row(feature_02="null")), ["records", 0, "feature_02"], "float_type"),
        (_records(_raw_row(feature_02="[1]")), ["records", 0, "feature_02"], "float_type"),
        (_records(_raw_row(site="false")), ["records", 0, "site"], "string_type"),
        (_records(_raw_row(site="1.5")), ["records", 0, "site"], "string_type"),
        (_records(_raw_row(), "{}"), ["records", 1, "feature_01"], "missing"),
        (_records(), ["records"], "too_short"),
        ('{"records": {"0": {}}}', ["records"], "list_type"),
        ('{"rows": []}', ["records"], "missing"),
        ("[]", [], "model_type"),
        ('{"records": [', [], "json_invalid"),
        ("", [], "json_invalid"),
    ],
)
def test_invalid_bodies_are_422_with_a_column_named_location(
    client: TestClient, ort_session: MagicMock, raw_body: str, loc: list[Any], error_type: str
) -> None:
    response = client.post(
        "/v1/predict", content=raw_body, headers={"Content-Type": "application/json"}
    )
    body = _assert_error(response, 422, "validation_error")
    # A union column (categorical str | int) adds the branch to the location, e.g.
    # ["records", 0, "site", "str"], so match on the column-level prefix.
    assert any(d["loc"][: len(loc)] == loc and d["type"] == error_type for d in body["details"]), (
        body["details"]
    )
    ort_session.run.assert_not_called()


def test_one_row_over_the_batch_limit_is_rejected(client: TestClient) -> None:
    body = _assert_error(
        _post(client, {"records": [_row()] * (MAX_ROWS + 1)}), 422, "validation_error"
    )
    assert body["details"][0]["type"] == "too_long"


def test_every_bad_row_is_reported_not_just_the_first(client: TestClient) -> None:
    records = [_row(feature_01="x"), _row(), _row(site=None)]
    body = _assert_error(_post(client, {"records": records}), 422, "validation_error")
    assert {tuple(d["loc"][:2]) for d in body["details"]} == {("records", 0), ("records", 2)}


def test_validation_errors_never_echo_the_payload(client: TestClient) -> None:
    response = _post(client, {"records": [_row(feature_01="secret-account-42")]})
    _assert_error(response, 422, "validation_error")
    assert "secret-account-42" not in response.text


# --- transport guards -------------------------------------------------------------------


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded"])
def test_a_non_json_content_type_is_415(client: TestClient, content_type: str | None) -> None:
    headers = {"Content-Type": content_type} if content_type else {}
    response = client.post(
        "/v1/predict", content=json.dumps({"records": [_row()]}), headers=headers
    )
    _assert_error(response, 415, "unsupported_media_type")


def test_json_with_a_charset_parameter_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        content=json.dumps({"records": [_row()]}),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200


def test_an_oversized_body_is_413_before_parsing(
    client: TestClient, ort_session: MagicMock
) -> None:
    padding = "x" * MAX_BODY
    response = _post(client, {"records": [_row(site=padding)]})
    _assert_error(response, 413, "payload_too_large")
    ort_session.run.assert_not_called()


def test_a_chunked_body_without_content_length_is_still_capped(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"records": [{"site": "'
        for _ in range(MAX_BODY // 512 + 1):
            yield b"y" * 512
        yield b'"}]}'

    response = client.post(
        "/v1/predict", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert "content-length" not in {k.lower() for k in response.request.headers}
    _assert_error(response, 413, "payload_too_large")


def test_unknown_routes_and_methods_use_the_error_shape(client: TestClient) -> None:
    _assert_error(client.get("/v2/predict"), 404, "not_found")
    _assert_error(client.get("/v1/predict"), 405, "method_not_allowed")


# --- failures inside the service --------------------------------------------------------


def test_a_runtime_failure_is_500_without_internals(
    client: TestClient, ort_session: MagicMock
) -> None:
    ort_session.run.side_effect = RuntimeError("[ONNXRuntimeError] node /gru2/GRU failed")
    response = _post(client, {"records": [_row()]})
    body = _assert_error(response, 500, "inference_error")
    assert "gru2" not in response.text
    assert body["message"] == "the model failed to score the request"


def test_a_non_finite_prediction_is_never_serialised(
    client: TestClient, ort_session: MagicMock
) -> None:
    ort_session.run.side_effect = lambda names, feeds: [
        np.full((len(feeds["features"]), 1), np.nan)
    ]
    _assert_error(_post(client, {"records": [_row()]}), 500, "inference_error")


def test_an_unexpected_exception_is_500_internal_error(
    client: TestClient, service: InferenceService, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(*args: Any, **kwargs: Any) -> Any:
        raise KeyError("a bug, with a stack trace nobody should see")

    monkeypatch.setattr(service, "predict", explode)
    response = _post(client, {"records": [_row()]})
    body = _assert_error(response, 500, "internal_error")
    assert "stack trace" not in response.text
    assert body["message"] == "internal error"


# --- OpenAPI ----------------------------------------------------------------------------


def test_openapi_documents_the_manifest_columns(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    body_ref = schema["paths"]["/v1/predict"]["post"]["requestBody"]["content"]["application/json"]
    assert body_ref["schema"] == {"$ref": "#/components/schemas/PredictRequest"}
    assert set(components["FeatureRow"]["required"]) == set(NUMERIC + CATEGORICAL)
    assert components["PredictRequest"]["properties"]["records"]["maxItems"] == MAX_ROWS

    refs = {
        value.removeprefix("#/components/schemas/")
        for value in _walk(schema)
        if isinstance(value, str) and value.startswith("#/components/schemas/")
    }
    assert refs <= set(components), f"dangling $refs: {refs - set(components)}"


def _walk(node: Any) -> Iterator[Any]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)
    else:
        yield node


# --- concurrency ------------------------------------------------------------------------


def test_scoring_runs_off_the_event_loop_and_is_bounded(
    settings: ServiceSettings, session: LoadedSession
) -> None:
    """Slow scoring must not stall health checks, and in-flight calls stay <= the limit."""
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def slow_run(names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.2)
        with lock:
            state["active"] -= 1
        return [feeds["features"].sum(axis=1)]

    ort = make_mock_ort_session(session.preprocessor.n_features, run=slow_run)
    service = InferenceService(session, OnnxModel(ort), max_concurrency=2)
    app = create_app(settings, service=service)

    async def scenario() -> tuple[list[int], float]:
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://api") as http,
        ):
            predictions = [
                asyncio.create_task(http.post("/v1/predict", json={"records": [_row()]}))
                for _ in range(6)
            ]
            await asyncio.sleep(0.05)
            started = time.perf_counter()
            live = await http.get("/health/live")
            live_latency = time.perf_counter() - started
            assert live.status_code == 200
            return [r.status_code for r in await asyncio.gather(*predictions)], live_latency

    statuses, live_latency = asyncio.run(scenario())
    assert statuses == [200] * 6
    assert state["peak"] == 2, state
    assert live_latency < 0.15, "a health check waited behind scoring: the loop was blocked"


# --- boot from settings, with a real ONNX file ------------------------------------------


def test_the_app_loads_its_session_from_settings_in_lifespan(
    settings: ServiceSettings, session: LoadedSession
) -> None:
    """No injected service: the lifespan loads the session and a real ONNX Runtime graph."""
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        records = [_row(), _row(site="south")]
        response = _post(client, {"records": records})
    assert response.status_code == 200, response.text
    # The real linear graph weights every feature by 1.0: the same row sum as the mock.
    np.testing.assert_allclose(
        response.json()["predictions"], _expected(session, records), rtol=1e-5
    )


def test_settings_are_read_from_the_environment(
    session: LoadedSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLPP_SESSION_DIR", str(session.session_dir))
    monkeypatch.setenv("MLPP_MAX_BATCH_ROWS", "2")
    with TestClient(create_app()) as client:
        assert client.get("/v1/model").json()["max_batch_rows"] == 2
        _assert_error(_post(client, {"records": [_row()] * 3}), 422, "validation_error")


def test_an_invalid_session_fails_at_boot_not_per_request(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        create_app(ServiceSettings(session_dir=tmp_path / "nowhere"))


def test_service_rejects_a_non_positive_concurrency_limit(
    session: LoadedSession, ort_session: MagicMock
) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        InferenceService(session, OnnxModel(ort_session), max_concurrency=0)


def test_contract_fixture_matches_the_manifest(session: LoadedSession) -> None:
    """Guards the fixture itself: the API tests assume this exact contract."""
    assert session.manifest.features == FeatureContract(
        NUMERIC, CATEGORICAL, "target", (*NUMERIC, "site_north", "site_south")
    )


# --- payload capture --------------------------------------------------------------------


def test_scored_payloads_are_captured_for_drift_analysis(
    session: LoadedSession, ort_session: MagicMock, settings: ServiceSettings, tmp_path: Path
) -> None:
    from mlpp.monitoring.payloads import JsonlPayloadSink, read_payloads

    sink = JsonlPayloadSink(tmp_path / "payloads")
    service = InferenceService(session, OnnxModel(ort_session), sink=sink)
    records = [_row(), _row(feature_01=-1.5, site="south")]
    with TestClient(create_app(settings, service=service)) as client:
        response = _post(client, {"records": records}, headers={REQUEST_ID_HEADER: "cap-1"})
        _post(client, {"records": [_row(feature_01="bad")]})  # rejected: never captured

    captured = read_payloads(sink.directory)
    assert captured[list(NUMERIC + CATEGORICAL)].to_dict("records") == records
    assert captured["prediction"].tolist() == response.json()["predictions"]
    assert set(captured["request_id"]) == {"cap-1"}


def test_a_failing_payload_sink_never_fails_the_request(
    session: LoadedSession,
    ort_session: MagicMock,
    settings: ServiceSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    broken = MagicMock()
    broken.write.side_effect = OSError("disk full")
    service = InferenceService(session, OnnxModel(ort_session), sink=broken)
    with TestClient(create_app(settings, service=service)) as client:
        response = _post(client, {"records": [_row()]})
    assert response.status_code == 200
    assert "payload capture failed" in caplog.text


def test_settings_enable_capture_through_the_payload_dir(
    session: LoadedSession, tmp_path: Path
) -> None:
    payload_dir = tmp_path / "captured"
    settings = ServiceSettings(session_dir=session.session_dir, payload_dir=payload_dir)
    with TestClient(create_app(settings)) as client:
        assert _post(client, {"records": [_row()]}).status_code == 200
    assert len(list(payload_dir.glob("payloads-*.jsonl"))) == 1

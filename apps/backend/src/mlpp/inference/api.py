"""Async FastAPI service over one trained session, scored through ONNX Runtime.

Endpoints:

- `GET  /health/live`  — the process is up (never touches the model);
- `GET  /health/ready` — the model is loaded and requests will be served;
- `GET  /v1/model`     — the column contract a client must satisfy;
- `POST /v1/predict`   — score 1..`max_batch_rows` records.

The request body model is generated from the session manifest at start-up
(`mlpp.schemas.inference.build_request_model`), so `/docs` lists the real feature
columns. Because the type only exists at runtime, the body is validated explicitly
with `model_validate_json` — pydantic-core parses and validates the bytes in one pass
— and the schema is spliced into the OpenAPI document by hand.

Every non-2xx response is an `ErrorResponse`, tracebacks never leave the process,
and every response carries an `X-Request-ID` (the caller's, if it is well-formed).

Run with `uvicorn mlpp.inference.api:create_app --factory` (settings from `MLPP_*`
environment variables) or `mlpp-serve --session <dir>` for local use.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from mlpp import __version__
from mlpp.errors import InferenceError, MlppError, SchemaError
from mlpp.inference.service import InferenceService
from mlpp.inference.settings import ServiceSettings
from mlpp.schemas.inference import (
    DEFAULT_MAX_ROWS,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    LivenessResponse,
    ModelInfo,
    PredictRequestBase,
    PredictResponse,
    ReadinessResponse,
    build_request_model,
    error_details,
)
from mlpp.session import FeatureContract, read_manifest

log = logging.getLogger(__name__)

REQUEST_ID_HEADER: Final = "X-Request-ID"
#: A caller-supplied request ID is echoed only if it is short and log-safe.
_SAFE_REQUEST_ID: Final = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_DEFAULT_MAX_BODY: Final = 5 * 1024 * 1024

_STATUS_CODES: Final[dict[int, ErrorCode]] = {
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    503: "model_unavailable",
}

_ERROR_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    status: {"model": ErrorResponse, "description": code.replace("_", " ")}
    for status, code in {
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        500: "internal_error",
        503: "model_unavailable",
    }.items()
}


class ApiError(Exception):
    """An error with a status and an `ErrorResponse` code, raised inside a route."""

    def __init__(
        self,
        status: int,
        code: ErrorCode,
        message: str,
        details: Sequence[ErrorDetail] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = list(details)


def create_app(
    settings: ServiceSettings | None = None, *, service: InferenceService | None = None
) -> FastAPI:
    """Build the app. With no arguments, settings come from the `MLPP_*` environment.

    `service` injects an already-built `InferenceService` (tests use one wrapping a
    mocked ONNX session); otherwise the session is loaded during lifespan start-up,
    in a worker thread so the loop is free, and `/health/ready` reports 503 until then.

    Reading the manifest here (not in lifespan) is deliberate: the request model is
    built from it, and a missing or invalid session should fail the process at boot
    rather than leave a live container that rejects every request.
    """
    if service is None and settings is None:
        settings = ServiceSettings()  # type: ignore[call-arg]  # session_dir from env
    contract: FeatureContract
    if service is not None:
        contract = service.contract
    else:
        assert settings is not None
        contract = read_manifest(settings.session_dir).features
    max_rows = settings.max_batch_rows if settings is not None else DEFAULT_MAX_ROWS
    max_body = settings.max_body_bytes if settings is not None else _DEFAULT_MAX_BODY
    request_model = build_request_model(contract, max_rows=max_rows)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.service = service
        else:
            assert settings is not None
            app.state.service = await asyncio.to_thread(InferenceService.load, settings)
            log.info("serving session %s", app.state.service.model_ref.session)
        yield
        app.state.service = None

    app = FastAPI(
        title="mlpp inference",
        version=__version__,
        description="CNN + Bi-GRU regressor served through ONNX Runtime.",
        lifespan=lifespan,
    )
    app.state.service = None
    _install_middleware(app)
    _install_error_handlers(app)

    @app.get("/health/live", response_model=LivenessResponse, tags=["health"])
    async def live() -> LivenessResponse:
        return LivenessResponse()

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: _ERROR_RESPONSES[503]},
        tags=["health"],
    )
    async def ready(request: Request) -> ReadinessResponse:
        return ReadinessResponse(model=_service(request).model_ref)

    @app.get(
        "/v1/model",
        response_model=ModelInfo,
        responses={503: _ERROR_RESPONSES[503]},
        tags=["model"],
    )
    async def model_info(request: Request) -> ModelInfo:
        return _service(request).model_info(max_batch_rows=max_rows)

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        responses=_ERROR_RESPONSES,
        tags=["model"],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/PredictRequest"}}
                },
            }
        },
    )
    async def predict(request: Request) -> PredictResponse:
        service = _service(request)
        _require_json(request)
        body = await _read_body(request, limit=max_body)
        payload = _parse(request_model, body)
        return await service.predict(payload, request_id=request.state.request_id)

    app.openapi = _openapi_with(app, request_model)  # type: ignore[method-assign]
    return app


# --- request plumbing -------------------------------------------------------------------


def _service(request: Request) -> InferenceService:
    service: InferenceService | None = request.app.state.service
    if service is None:
        raise ApiError(503, "model_unavailable", "the model is not loaded yet")
    return service


def _require_json(request: Request) -> None:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise ApiError(
            415,
            "unsupported_media_type",
            f"expected Content-Type application/json, got {media_type or 'none'!r}",
        )


async def _read_body(request: Request, *, limit: int) -> bytes:
    """Read the body, refusing it as soon as it exceeds `limit` bytes.

    Checks `Content-Length` first, then counts while streaming, so a chunked upload
    without a length header cannot grow past the cap in memory either.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _too_large(limit)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _too_large(limit)
        chunks.append(chunk)
    return b"".join(chunks)


def _too_large(limit: int) -> ApiError:
    return ApiError(413, "payload_too_large", f"request body exceeds {limit} bytes")


def _parse(model: type[PredictRequestBase], body: bytes) -> PredictRequestBase:
    try:
        return model.model_validate_json(body)
    except ValidationError as exc:
        raise ApiError(
            422,
            "validation_error",
            f"request body failed validation ({exc.error_count()} error(s))",
            error_details(exc.errors(include_url=False, include_input=False)),
        ) from exc


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        rid = supplied if _SAFE_REQUEST_ID.match(supplied) else uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


# --- errors -----------------------------------------------------------------------------


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(request, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(
            request,
            422,
            "validation_error",
            "request failed validation",
            error_details(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "internal_error")
        return _error(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(SchemaError)
    async def schema_error(request: Request, exc: SchemaError) -> JSONResponse:
        return _error(request, 422, "schema_error", str(exc))

    @app.exception_handler(InferenceError)
    async def inference_error(request: Request, exc: InferenceError) -> JSONResponse:
        log.error("inference failed (request %s): %s", _rid(request), exc)
        return _error(request, 500, "inference_error", "the model failed to score the request")

    @app.exception_handler(MlppError)
    async def mlpp_error(request: Request, exc: MlppError) -> JSONResponse:
        log.error("pipeline error (request %s): %s", _rid(request), exc)
        return _error(request, 500, "internal_error", "internal error")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error (request %s)", _rid(request))
        return _error(request, 500, "internal_error", "internal error")


def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error(
    request: Request,
    status: int,
    code: ErrorCode,
    message: str,
    details: Sequence[ErrorDetail] = (),
) -> JSONResponse:
    rid = _rid(request)
    body = ErrorResponse(error=code, message=message, request_id=rid, details=list(details))
    headers = {REQUEST_ID_HEADER: rid} if rid else None
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"), headers=headers)


# --- OpenAPI ----------------------------------------------------------------------------


def _openapi_with(
    app: FastAPI, request_model: type[PredictRequestBase]
) -> Callable[[], dict[str, Any]]:
    """`app.openapi` replacement that registers the runtime-built request schema."""

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        body = request_model.model_json_schema(ref_template="#/components/schemas/{model}")
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        components.update(body.pop("$defs", {}))
        components["PredictRequest"] = body
        app.openapi_schema = schema
        return schema

    return openapi


# --- local entry point ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """`mlpp-serve` — run the API locally against a session directory."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="mlpp-serve", description=main.__doc__)
    parser.add_argument("--session", type=Path, required=True, help="session directory to serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    os.environ["MLPP_SESSION_DIR"] = str(args.session.resolve())
    uvicorn.run(
        "mlpp.inference.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

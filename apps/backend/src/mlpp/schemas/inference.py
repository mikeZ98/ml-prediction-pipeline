"""Strict Pydantic v2 contract for the inference API's JSON bodies.

The feature columns are not known at import time: they live in the session's
`manifest.json`, the single source of truth for the column contract. So the row
and request models are *built* from a `FeatureContract` once, at service start,
rather than hand-written — restating the columns here would give the contract a
second home that could drift from the manifest.

Strictness is the point. The preprocessor tolerates a lot (NaNs are mean-filled,
unknown categorical levels are zero-encoded), which is right for a training CSV
and wrong for a request: a single-row batch has no mean to fill from, so a `null`
silently becomes a NaN prediction. Every gap is therefore rejected at the edge:

- every input column is required and non-null; unknown keys are rejected;
- numeric values must be finite JSON numbers (no `NaN`, no `Infinity`, no `"1.5"`
  strings, no booleans);
- categorical values are strings or integers, never booleans.

Columns are attached as field *aliases*, not field names, so a column called
`json`, `model_config` or `flow rate` cannot collide with a BaseModel attribute or
fail identifier rules.

Imports nothing heavier than `mlpp.session`, so the schemas are testable in the
fast suite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, create_model

from mlpp.session import FeatureContract

#: Upper bound on records per request unless the service configures another.
DEFAULT_MAX_ROWS: Final = 1000

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
CategoricalValue = StrictStr | StrictInt
FeatureValue = float | int | str


class StrictModel(BaseModel):
    """Base for every wire model: no coercion, no unknown keys, immutable."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class FeatureRowBase(StrictModel):
    """One inference record. `build_row_model` subclasses it per contract."""

    def to_record(self) -> dict[str, FeatureValue]:
        """The record keyed by *column* name (the aliases), ready for a DataFrame."""
        return self.model_dump(by_alias=True)


class PredictRequestBase(StrictModel):
    """`POST /v1/predict` body. `build_request_model` narrows `records` per contract."""

    records: list[FeatureRowBase]

    def to_records(self) -> list[dict[str, FeatureValue]]:
        return [row.to_record() for row in self.records]


def build_row_model(contract: FeatureContract) -> type[FeatureRowBase]:
    """A strict model with one required field per input column of `contract`.

    Field names are positional (`c0`, `c1`, …) and the column is the alias, so the
    JSON keys, validation-error locations and the JSON schema all speak in column
    names while the Python side never has to.
    """
    columns = contract.numeric_columns + contract.categorical_columns
    if not columns:
        raise ValueError("the feature contract has no input columns")
    if len(set(columns)) != len(columns):
        raise ValueError(f"the feature contract repeats a column: {list(columns)}")

    fields: dict[str, Any] = {}
    for index, column in enumerate(contract.numeric_columns):
        fields[f"c{index}"] = (FiniteFloat, Field(alias=column))
    offset = len(contract.numeric_columns)
    for index, column in enumerate(contract.categorical_columns, start=offset):
        fields[f"c{index}"] = (CategoricalValue, Field(alias=column))
    return create_model("FeatureRow", __base__=FeatureRowBase, **fields)


def build_request_model(
    contract: FeatureContract, *, max_rows: int = DEFAULT_MAX_ROWS
) -> type[PredictRequestBase]:
    """The request body model: 1..`max_rows` records of `build_row_model(contract)`."""
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    row = build_row_model(contract)
    records = Annotated[list[row], Field(min_length=1, max_length=max_rows)]  # type: ignore[valid-type]
    return create_model("PredictRequest", __base__=PredictRequestBase, records=(records, ...))


class ModelRef(StrictModel):
    """Which trained session answered. The session directory name is its identity."""

    session: str
    created: str
    schema_version: int


class PredictResponse(StrictModel):
    request_id: str
    model: ModelRef
    n_records: int = Field(ge=1)
    #: One value per record, in request order, in the target's original units.
    predictions: list[float]
    #: Categorical column -> levels never seen in training. Those rows still score
    #: (encoded as all-zeros), which is why they are reported rather than rejected.
    unseen_levels: dict[str, list[str | int]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RuntimeInfo(StrictModel):
    """What the ONNX graph declares about itself."""

    input_name: str
    output_name: str
    opset: int | None
    producer: str
    n_parameters: int | None


class ModelInfo(StrictModel):
    """`GET /v1/model`: the contract a client must satisfy, straight from the manifest."""

    model: ModelRef
    output_column: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    #: Post-one-hot names, in the order they occupy the model's feature axis.
    feature_names: list[str]
    max_batch_rows: int
    runtime: RuntimeInfo


class LivenessResponse(StrictModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(StrictModel):
    status: Literal["ready"] = "ready"
    model: ModelRef


ErrorCode = Literal[
    "validation_error",
    "schema_error",
    "unsupported_media_type",
    "payload_too_large",
    "not_found",
    "method_not_allowed",
    "model_unavailable",
    "inference_error",
    "internal_error",
]


class ErrorDetail(StrictModel):
    """One validation failure. `loc` uses column names, e.g. `["records", 0, "feature_01"]`."""

    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(StrictModel):
    """Every non-2xx body has this shape — validation, runtime and unexpected failures alike."""

    error: ErrorCode
    message: str
    request_id: str | None = None
    details: list[ErrorDetail] = Field(default_factory=list)


def error_details(errors: Sequence[Mapping[str, Any]]) -> list[ErrorDetail]:
    """Project Pydantic/FastAPI error dicts onto `ErrorDetail`.

    Only `loc`, `msg` and `type` survive. The offending `input` is dropped on
    purpose: echoing a payload back in an error body leaks it into client logs and
    can be arbitrarily large.
    """
    return [
        ErrorDetail(
            loc=[part if isinstance(part, int) else str(part) for part in error.get("loc", ())],
            msg=str(error.get("msg", "")),
            type=str(error.get("type", "")),
        )
        for error in errors
    ]

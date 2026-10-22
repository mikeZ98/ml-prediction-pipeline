"""Strict inference schemas. Pure Pydantic — part of the fast suite."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from mlpp.schemas.inference import (
    ErrorResponse,
    PredictRequestBase,
    PredictResponse,
    build_request_model,
    build_row_model,
    error_details,
)
from mlpp.session import FeatureContract

NUMERIC = ("feature_01", "feature_02", "comp_active")
CATEGORICAL = ("site",)


@pytest.fixture
def contract() -> FeatureContract:
    return FeatureContract(
        numeric_columns=NUMERIC,
        categorical_columns=CATEGORICAL,
        output_column="target",
        feature_names=(*NUMERIC, "site_a", "site_b"),
    )


@pytest.fixture
def request_model(contract: FeatureContract) -> type[PredictRequestBase]:
    return build_request_model(contract, max_rows=3)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"feature_01": 0.5, "feature_02": -1.25, "comp_active": 1, "site": "a"}
    row.update(overrides)
    return row


def _parse(model: type[PredictRequestBase], body: str) -> PredictRequestBase:
    return model.model_validate_json(body)


def _errors(model: type[PredictRequestBase], body: str) -> list[dict[str, Any]]:
    with pytest.raises(ValidationError) as excinfo:
        _parse(model, body)
    return [dict(e) for e in excinfo.value.errors()]


def test_a_well_formed_request_round_trips_to_column_keyed_records(
    request_model: type[PredictRequestBase],
) -> None:
    parsed = _parse(request_model, json.dumps({"records": [_row(), _row(site=7)]}))
    assert parsed.to_records() == [_row(), _row(site=7)]


def test_json_integers_are_accepted_for_numeric_columns(
    request_model: type[PredictRequestBase],
) -> None:
    """`comp_active` is a 0/1 flag; strict mode must not demand `1.0`."""
    parsed = _parse(request_model, json.dumps({"records": [_row(feature_01=3)]}))
    assert parsed.to_records()[0]["feature_01"] == 3


@pytest.mark.parametrize(
    ("column", "raw", "error_type"),
    [
        ("feature_01", '"1.5"', "float_type"),
        ("feature_01", "true", "float_type"),
        ("feature_01", "null", "float_type"),
        ("feature_01", "NaN", "finite_number"),
        ("feature_01", "Infinity", "finite_number"),
        ("feature_01", "-Infinity", "finite_number"),
        ("feature_01", "1e400", "finite_number"),
        ("feature_01", "[1.0]", "float_type"),
        ("site", "true", "string_type"),
        ("site", "null", "string_type"),
        ("site", "1.5", "string_type"),
    ],
)
def test_values_the_preprocessor_would_silently_absorb_are_rejected(
    request_model: type[PredictRequestBase], column: str, raw: str, error_type: str
) -> None:
    row = json.dumps(_row()).replace(
        f'"{column}": {json.dumps(_row()[column])}', f'"{column}": {raw}'
    )
    errors = _errors(request_model, f'{{"records": [{row}]}}')
    locations = {tuple(e["loc"][:3]) for e in errors}
    assert ("records", 0, column) in locations, errors
    assert any(e["type"] == error_type for e in errors), errors


def test_a_missing_column_is_reported_by_its_column_name(
    request_model: type[PredictRequestBase],
) -> None:
    row = _row()
    del row["feature_02"]
    errors = _errors(request_model, json.dumps({"records": [_row(), row]}))
    assert [(tuple(e["loc"]), e["type"]) for e in errors] == [
        (("records", 1, "feature_02"), "missing")
    ]


def test_an_unknown_column_is_rejected_not_ignored(
    request_model: type[PredictRequestBase],
) -> None:
    errors = _errors(request_model, json.dumps({"records": [_row(feature_99=1.0)]}))
    assert [(tuple(e["loc"]), e["type"]) for e in errors] == [
        (("records", 0, "feature_99"), "extra_forbidden")
    ]


def test_the_target_column_is_not_an_accepted_input(
    request_model: type[PredictRequestBase],
) -> None:
    """Sending the truth column is a client bug; it must not reach the model silently."""
    errors = _errors(request_model, json.dumps({"records": [_row(target=1.0)]}))
    assert errors[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("records", "error_type"),
    [([], "too_short"), ([_row()] * 4, "too_long"), ({"0": _row()}, "list_type")],
)
def test_batch_bounds_are_enforced(
    request_model: type[PredictRequestBase], records: Any, error_type: str
) -> None:
    errors = _errors(request_model, json.dumps({"records": records}))
    assert errors[0]["type"] == error_type
    assert errors[0]["loc"] == ("records",)


@pytest.mark.parametrize(
    ("body", "error_type"),
    [
        ("{}", "missing"),
        ('{"records": [], "extra": 1}', "extra_forbidden"),
        ("[]", "model_type"),
        ('{"records": [', "json_invalid"),
        ("", "json_invalid"),
    ],
)
def test_malformed_envelopes_are_rejected(
    request_model: type[PredictRequestBase], body: str, error_type: str
) -> None:
    assert any(e["type"] == error_type for e in _errors(request_model, body))


def test_columns_that_shadow_basemodel_attributes_or_are_not_identifiers_work() -> None:
    awkward = ("json", "model_config", "flow rate", "1st")
    contract = FeatureContract(awkward, (), "target", awkward)
    model = build_request_model(contract)
    record = {name: float(i) for i, name in enumerate(awkward)}
    assert model.model_validate_json(json.dumps({"records": [record]})).to_records() == [record]


def test_the_json_schema_names_every_column_as_required(contract: FeatureContract) -> None:
    schema = build_row_model(contract).model_json_schema()
    assert set(schema["required"]) == set(NUMERIC + CATEGORICAL)
    assert schema["additionalProperties"] is False


def test_a_contract_without_inputs_cannot_build_a_model() -> None:
    with pytest.raises(ValueError, match="no input columns"):
        build_row_model(FeatureContract((), (), "target", ()))


def test_max_rows_must_be_positive(contract: FeatureContract) -> None:
    with pytest.raises(ValueError, match="max_rows"):
        build_request_model(contract, max_rows=0)


def test_error_details_drop_the_offending_input(
    request_model: type[PredictRequestBase],
) -> None:
    """Echoing a payload back in an error body leaks it into client logs."""
    with pytest.raises(ValidationError) as excinfo:
        _parse(request_model, json.dumps({"records": [_row(feature_01="secret-value")]}))
    details = error_details(excinfo.value.errors())
    assert details[0].loc == ["records", 0, "feature_01"]
    assert "secret-value" not in json.dumps([d.model_dump() for d in details])


def test_error_response_rejects_an_unknown_code() -> None:
    with pytest.raises(ValidationError):
        ErrorResponse(error="teapot", message="no")  # type: ignore[arg-type]


def test_predict_response_is_strict_about_its_own_types() -> None:
    """The response is a contract too: a stray string prediction must not serialise."""
    with pytest.raises(ValidationError):
        PredictResponse(
            request_id="r",
            model={"session": "s", "created": "c", "schema_version": 2},  # type: ignore[arg-type]
            n_records=1,
            predictions=["1.0"],  # type: ignore[list-item]
        )

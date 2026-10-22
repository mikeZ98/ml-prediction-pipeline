"""The scoring service behind the HTTP layer: one loaded session, scored off the event loop.

Framework-agnostic on purpose — it takes and returns schema objects, so the API is a
thin adapter and this class is testable without HTTP. Scoring goes through the same
`mlpp.predict.score_frame` the CLI and dashboard use, so the three cannot disagree
about preprocessing or the inverse target transform.

Scoring is CPU-bound and ONNX Runtime releases the GIL, so each call runs in a worker
thread (`asyncio.to_thread`) and the event loop keeps serving health checks and
other requests. A semaphore bounds how many calls are in flight.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd

from mlpp.errors import InferenceError
from mlpp.inference.onnx_model import OnnxModel
from mlpp.inference.settings import ServiceSettings
from mlpp.monitoring.payloads import JsonlPayloadSink, PayloadSink
from mlpp.predict import PredictionResult, load_model, score_frame
from mlpp.predict_cli import columns_from
from mlpp.schemas.inference import (
    ModelInfo,
    ModelRef,
    PredictRequestBase,
    PredictResponse,
    RuntimeInfo,
)
from mlpp.session import FeatureContract, LoadedSession, load_session

log = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        session: LoadedSession,
        model: OnnxModel,
        *,
        max_concurrency: int = 4,
        sink: PayloadSink | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._session = session
        self._model = model
        self._sink = sink
        self._limit = asyncio.Semaphore(max_concurrency)
        features = session.manifest.features
        self._columns = list(features.numeric_columns + features.categorical_columns)
        self._ref = ModelRef(
            # From the manifest, not the path: in a container the directory is a
            # mount point (/models/session) and its name says nothing about the run.
            session=session.manifest.session_id,
            created=session.manifest.created,
            schema_version=session.manifest.schema_version,
        )

    @classmethod
    def load(cls, settings: ServiceSettings) -> InferenceService:
        """Restore the session and its ONNX model. Blocking — call it off the event loop.

        The schema is always strict here: a request missing a column is rejected by
        the request model long before this point, and lenient zero-filling would turn
        a client bug into a plausible-looking prediction.
        """
        columns = columns_from(settings.session_dir, strict_schema=True)
        session = load_session(settings.session_dir, columns)
        model = load_model(session, intra_op_threads=settings.intra_op_threads)
        sink = JsonlPayloadSink(settings.payload_dir) if settings.payload_dir else None
        return cls(session, model, max_concurrency=settings.max_concurrency, sink=sink)

    @property
    def contract(self) -> FeatureContract:
        return self._session.manifest.features

    @property
    def model_ref(self) -> ModelRef:
        return self._ref

    def model_info(self, *, max_batch_rows: int) -> ModelInfo:
        features = self.contract
        description = self._model.describe()
        return ModelInfo(
            model=self._ref,
            output_column=features.output_column,
            numeric_columns=list(features.numeric_columns),
            categorical_columns=list(features.categorical_columns),
            feature_names=list(features.feature_names),
            max_batch_rows=max_batch_rows,
            runtime=RuntimeInfo(
                input_name=description.input_name,
                output_name=description.output_name,
                opset=description.opset,
                producer=description.producer,
                n_parameters=description.n_parameters,
            ),
        )

    async def predict(self, request: PredictRequestBase, *, request_id: str) -> PredictResponse:
        """Score every record in `request`, in order, in the target's original units.

        Raises InferenceError when the runtime fails or returns non-finite values — a
        NaN cannot be represented in a JSON response, and a client should never have
        to guess whether one means "no prediction".
        """
        records = request.to_records()
        frame = pd.DataFrame.from_records(records, columns=self._columns)
        async with self._limit:
            result = await asyncio.to_thread(score_frame, self._session, self._model, frame)

        predictions = np.asarray(result.predictions, dtype=np.float64)
        if not np.isfinite(predictions).all():
            bad = int((~np.isfinite(predictions)).sum())
            raise InferenceError(f"the model produced {bad} non-finite prediction(s)")
        values = predictions.tolist()
        await self._capture(records, values, request_id)
        return PredictResponse(
            request_id=request_id,
            model=self._ref,
            n_records=len(records),
            predictions=values,
            unseen_levels=_json_levels(result),
            warnings=[result.describe_unseen()] if result.has_unseen else [],
        )

    async def _capture(
        self, records: list[dict[str, float | int | str]], predictions: list[float], rid: str
    ) -> None:
        """Hand the scored batch to the payload sink, off the event loop.

        Capture is best-effort by design: monitoring must never turn a successful
        prediction into a failed request, so a sink error is logged and swallowed.
        """
        if self._sink is None:
            return
        try:
            await asyncio.to_thread(
                self._sink.write, records, predictions, request_id=rid, session=self._ref.session
            )
        except Exception:
            log.exception("payload capture failed for request %s; prediction still returned", rid)


def _json_levels(result: PredictionResult) -> dict[str, list[str | int]]:
    """Unseen levels as plain JSON scalars (numpy ints would fail strict validation)."""
    return {
        column: [_scalar(value) for value in values] for column, values in result.unseen.items()
    }


def _scalar(value: object) -> str | int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return str(value)

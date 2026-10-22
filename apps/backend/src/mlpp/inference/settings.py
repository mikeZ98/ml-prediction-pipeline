"""Service configuration, read from `MLPP_*` environment variables.

Twelve-factor style: the container image is generic and the session it serves,
its limits and its thread budget are injected at run time.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlpp.schemas.inference import DEFAULT_MAX_ROWS


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLPP_", frozen=True, extra="ignore")

    #: Trained session directory to serve (must contain manifest.json).
    session_dir: Path
    #: Records accepted per request; larger batches get 422.
    max_batch_rows: int = Field(DEFAULT_MAX_ROWS, ge=1, le=100_000)
    #: Request bodies above this many bytes get 413 before they are parsed.
    max_body_bytes: int = Field(5 * 1024 * 1024, ge=1024)
    #: Scoring calls allowed in flight at once. Each runs in a worker thread, and ONNX
    #: Runtime parallelises inside a call, so this caps CPU oversubscription.
    max_concurrency: int = Field(4, ge=1, le=256)
    #: ONNX Runtime intra-op threads per call; None lets the runtime decide.
    intra_op_threads: int | None = Field(None, ge=1)
    #: Where scored payloads are captured for drift analysis; None disables capture.
    payload_dir: Path | None = None

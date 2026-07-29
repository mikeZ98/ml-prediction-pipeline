"""Feature schema resolution, scaling and one-hot encoding.

The notebook re-derived the numeric/categorical split from every file it touched,
so a test CSV whose dtypes inferred differently produced a different feature
order — silently misaligned with the trained model. Here the split is resolved
once during `fit`, stored on the Preprocessor, and reused by every `transform`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlpp.config import DataConfig
from mlpp.errors import ArtifactError, NotFittedError, SchemaError

SCALER_FILE = "scaler.gz"
OUTPUT_SCALER_FILE = "output_scaler.gz"
ENCODERS_FILE = "encoders.gz"


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """The column split, frozen at fit time."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


def resolve_schema(df: pd.DataFrame, cfg: DataConfig) -> FeatureSchema:
    """Split `cfg.input_columns` into numeric and categorical using `df`'s dtypes.

    Columns named in `cfg.categorical_columns` are categorical regardless of dtype.
    """
    forced = set(cfg.categorical_columns)
    numeric, categorical = [], []
    for col in cfg.input_columns:
        if col in forced or (col in df.columns and not pd.api.types.is_numeric_dtype(df[col])):
            categorical.append(col)
        else:
            numeric.append(col)
    return FeatureSchema(tuple(numeric), tuple(categorical), cfg.output_column)


def align_columns(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Return a copy holding exactly the configured columns, in configured order.

    Missing inputs raise SchemaError under `strict_schema`, else are filled with 0.0.
    The caller's frame is never mutated.
    """
    missing = [c for c in cfg.input_columns if c not in df.columns]
    if missing and cfg.strict_schema:
        raise SchemaError(f"missing input columns: {missing}")

    aligned = df.copy()
    for col in missing:
        aligned[col] = 0.0

    keep = list(cfg.input_columns)
    if cfg.output_column in aligned.columns:
        keep.append(cfg.output_column)
    return aligned[keep]


def _fill_missing(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Numeric NaNs -> column mean; categorical NaNs -> forward then backward fill."""
    filled = df.copy()
    for col in schema.numeric:
        if filled[col].isna().any():
            filled[col] = filled[col].fillna(filled[col].mean())
    for col in schema.categorical:
        if filled[col].isna().any():
            # .ffill()/.bfill(); DataFrame.fillna(method=...) was removed in pandas 3.
            filled[col] = filled[col].astype("object").ffill().bfill()
    return filled


def _new_onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


class Preprocessor:
    """Fits scalers/encoders once, then transforms every later frame identically."""

    def __init__(self, cfg: DataConfig, *, use_onehot: bool = True) -> None:
        self._cfg = cfg
        self._use_onehot = use_onehot
        self._scaler = StandardScaler()
        self._output_scaler = StandardScaler()
        self._onehot = _new_onehot()
        self._schema: FeatureSchema | None = None
        self._feature_names: tuple[str, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return self._schema is not None

    @property
    def schema(self) -> FeatureSchema:
        if self._schema is None:
            raise NotFittedError("Preprocessor.fit has not been called")
        return self._schema

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Model-input column names, in the order they occupy axis 1 of X.

        Numeric columns keep their names; categoricals expand to one-hot names.
        Use this — not the raw config — to locate a column inside X.
        """
        if self._schema is None:
            raise NotFittedError("Preprocessor.fit has not been called")
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def index_of(self, column: str) -> int:
        """Position of `column` on axis 1 of X. Raises KeyError if it is not a feature."""
        try:
            return self.feature_names.index(column)
        except ValueError as exc:
            raise KeyError(f"{column!r} is not a model feature") from exc

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Learn the schema, scalers and encoder categories from `df`."""
        aligned = align_columns(df, self._cfg)
        schema = resolve_schema(aligned, self._cfg)
        filled = _fill_missing(aligned, schema)

        if schema.numeric:
            self._scaler.fit(filled[list(schema.numeric)])
        if self._encode_categoricals(schema):
            self._onehot.fit(filled[list(schema.categorical)])

        if schema.output not in filled.columns:
            raise SchemaError(f"target column {schema.output!r} required to fit")
        self._output_scaler.fit(filled[[schema.output]].to_numpy(dtype=np.float64))

        self._schema = schema
        self._feature_names = self._build_feature_names(schema)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Return `(X, y)`; `y` is None when `df` carries no target column.

        X has shape (n_rows, n_features, 1) — Conv1D runs across the feature axis.
        """
        schema = self.schema
        aligned = align_columns(df, self._cfg)
        filled = _fill_missing(aligned, schema)

        blocks: list[np.ndarray] = []
        if schema.numeric:
            blocks.append(self._scaler.transform(filled[list(schema.numeric)]))
        if self._encode_categoricals(schema):
            blocks.append(np.asarray(self._onehot.transform(filled[list(schema.categorical)])))

        x = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else _empty(len(filled))
        if x.shape[1] != self.n_features:
            msg = f"produced {x.shape[1]} features, model expects {self.n_features}"
            raise SchemaError(msg)
        x = np.expand_dims(x, axis=-1)

        y = None
        if schema.output in filled.columns:
            y = self._output_scaler.transform(
                filled[[schema.output]].to_numpy(dtype=np.float64)
            ).astype(np.float32)
        return x, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        return self.fit(df).transform(df)

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        """Map scaled targets/predictions back to original units, as a 1-D array."""
        if self._schema is None:
            raise NotFittedError("Preprocessor.fit has not been called")
        restored = self._output_scaler.inverse_transform(np.asarray(y).reshape(-1, 1))
        return np.asarray(restored).ravel()

    def _encode_categoricals(self, schema: FeatureSchema) -> bool:
        return self._use_onehot and bool(schema.categorical)

    def _build_feature_names(self, schema: FeatureSchema) -> tuple[str, ...]:
        names = list(schema.numeric)
        if self._encode_categoricals(schema):
            names += [str(n) for n in self._onehot.get_feature_names_out(schema.categorical)]
        return tuple(names)

    def save(self, out_dir: Path) -> None:
        """Persist scalers and encoders next to the model artifacts."""
        if self._schema is None:
            raise NotFittedError("refusing to save an unfitted Preprocessor")
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, out_dir / SCALER_FILE)
        joblib.dump(self._output_scaler, out_dir / OUTPUT_SCALER_FILE)
        joblib.dump(
            {"onehot": self._onehot, "schema": self._schema, "feature_names": self._feature_names},
            out_dir / ENCODERS_FILE,
        )

    @classmethod
    def load(cls, out_dir: Path, cfg: DataConfig, *, use_onehot: bool = True) -> Preprocessor:
        """Restore a Preprocessor previously written by `save`."""
        pre = cls(cfg, use_onehot=use_onehot)
        for name in (SCALER_FILE, OUTPUT_SCALER_FILE, ENCODERS_FILE):
            if not (out_dir / name).is_file():
                raise ArtifactError(f"missing artifact {name} in {out_dir}")
        pre._scaler = joblib.load(out_dir / SCALER_FILE)
        pre._output_scaler = joblib.load(out_dir / OUTPUT_SCALER_FILE)
        bundle: dict[str, Any] = joblib.load(out_dir / ENCODERS_FILE)
        try:
            pre._onehot = bundle["onehot"]
            pre._schema = bundle["schema"]
            pre._feature_names = tuple(bundle["feature_names"])
        except KeyError as exc:
            raise ArtifactError(f"{ENCODERS_FILE} is missing key {exc}") from exc
        return pre


def _empty(n_rows: int) -> np.ndarray:
    return np.empty((n_rows, 0), dtype=np.float32)

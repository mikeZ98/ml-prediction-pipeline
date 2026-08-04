"""`mlpp-predict` entrypoint — score a new CSV against a trained session.

Reads the column contract from the session's own `manifest.json`, so scoring needs
nothing but the session directory and an input file. Predictions are written to a
path the caller names: a session directory is a reproducible training artifact and
this command never writes into one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from mlpp.config import ColumnConfig
from mlpp.data import read_table_auto
from mlpp.errors import MlppError
from mlpp.session import LoadedSession, load_session, read_manifest

log = logging.getLogger(__name__)

PREDICTION_COLUMN = "prediction"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlpp-predict", description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="session directory produced by mlpp-train (must contain manifest.json)",
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV or Parquet file to score")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="where to write predictions (never inside the session directory)",
    )
    parser.add_argument(
        "--lenient-schema",
        action="store_true",
        help="fill missing input columns with 0.0 instead of failing",
    )
    parser.add_argument(
        "--keep-inputs",
        action="store_true",
        help="write the input columns alongside the prediction column",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress Keras progress bars")
    return parser


def columns_from(session_dir: Path, *, strict_schema: bool) -> ColumnConfig:
    """Rebuild the column contract from the session's manifest.

    The manifest is the single source of truth for the column contract, so the
    caller never has to restate what the model was trained on. `strict_schema` is
    policy rather than a persisted fact, which is why it stays a flag.
    """
    features = read_manifest(session_dir).features
    return ColumnConfig(
        input_columns=features.numeric_columns + features.categorical_columns,
        output_column=features.output_column,
        categorical_columns=features.categorical_columns,
        strict_schema=strict_schema,
    )


def prediction_frame(
    source: pd.DataFrame, predictions: pd.Series, *, keep_inputs: bool
) -> pd.DataFrame:
    """Shape predictions for output: the prediction column alone, or beside the inputs.

    Public and shared with the dashboard's batch panel. Duplicating this would give
    the CSV a different column layout depending on whether it came from the CLI or a
    download button — the kind of quiet divergence `session.py` exists to prevent.

    Defined in this module because it is TensorFlow-free (Keras is imported inside
    `_score`), so a caller can import it without paying the training stack.
    """
    if not keep_inputs:
        return predictions.to_frame()
    out = source.copy()
    out[PREDICTION_COLUMN] = predictions.to_numpy()
    return out


def _score(session: LoadedSession, frame: pd.DataFrame) -> pd.Series:
    """Score `frame`, warning on unseen levels without failing the run.

    `mlpp.predict` is imported here rather than at module scope so `--help` and
    argument errors never pay the multi-second TensorFlow import.
    """
    from mlpp.predict import load_model, score_frame

    result = score_frame(session, load_model(session), frame)
    if result.has_unseen:
        print(f"warning: {result.describe_unseen()}", file=sys.stderr)
    return pd.Series(result.predictions, index=frame.index, name=PREDICTION_COLUMN)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        columns = columns_from(args.session, strict_schema=not args.lenient_schema)
        session = load_session(args.session, columns)
        frame = read_table_auto(args.input)
        predictions = _score(session, frame)
        out = prediction_frame(frame, predictions, keep_inputs=args.keep_inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
    except MlppError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"predictions -> {args.output} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

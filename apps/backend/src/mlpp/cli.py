"""`mlpp-train` entrypoint — runs the pipeline without opening a notebook."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from mlpp.config import (
    DEFAULT_ACTIVE_COLUMN,
    DEFAULT_INPUT_COLUMNS,
    DEFAULT_OUTPUT_COLUMN,
    ColumnConfig,
    DataConfig,
    EvalConfig,
    PipelineConfig,
    TrackingConfig,
    TrainConfig,
)
from mlpp.errors import MlppError
from mlpp.pipeline import run_pipeline

DEFAULT_ROOT = Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlpp-train", description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="repo root holding TRAIN/ TEST/ OUTPUTS/"
    )
    parser.add_argument("--train-dir", type=Path, help="override TRAIN directory")
    parser.add_argument("--test-dir", type=Path, help="override TEST directory")
    parser.add_argument("--output-dir", type=Path, help="override OUTPUTS directory")
    parser.add_argument(
        "--input-columns",
        nargs="+",
        metavar="COL",
        default=list(DEFAULT_INPUT_COLUMNS),
        help="feature columns, in order (default: feature_01..feature_12 comp_active)",
    )
    parser.add_argument("--output-column", default=DEFAULT_OUTPUT_COLUMN, help="target column")
    parser.add_argument(
        "--categorical-columns",
        nargs="*",
        metavar="COL",
        default=[],
        help="subset of --input-columns to one-hot encode regardless of dtype",
    )
    parser.add_argument(
        "--lenient-schema",
        action="store_true",
        help="fill missing input columns with 0.0 instead of failing",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--active-weight-alpha", type=float, default=0.0)
    parser.add_argument("--rolling-window", type=int, default=500)
    parser.add_argument("--no-plots", action="store_true", help="skip PNG/HTML artifacts")
    tracking = parser.add_argument_group("experiment tracking (MLflow)")
    tracking.add_argument(
        "--no-tracking", action="store_true", help="do not record the run in MLflow"
    )
    tracking.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI, else a local SQLite "
        "store under <root>/mlruns/)",
    )
    tracking.add_argument("--experiment", default="mlpp", help="MLflow experiment name")
    tracking.add_argument(
        "--register-model",
        metavar="NAME",
        help="also register the ONNX model in the MLflow Model Registry under NAME",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-epoch progress logs")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="training device; auto picks CUDA when available, else CPU",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    root: Path = args.root
    return PipelineConfig(
        data=DataConfig(
            train_dir=args.train_dir or root / "TRAIN",
            test_dir=args.test_dir or root / "TEST",
            output_dir=args.output_dir or root / "OUTPUTS",
            columns=ColumnConfig(
                input_columns=tuple(args.input_columns),
                output_column=args.output_column,
                categorical_columns=tuple(args.categorical_columns),
                strict_schema=not args.lenient_schema,
            ),
        ),
        train=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            loss=args.loss,
            device=args.device,
            active_weight_alpha=args.active_weight_alpha,
            active_column=(
                DEFAULT_ACTIVE_COLUMN if DEFAULT_ACTIVE_COLUMN in args.input_columns else None
            ),
        ),
        eval=EvalConfig(rolling_window=args.rolling_window, write_plots=not args.no_plots),
        tracking=tracking_from_args(args, root),
    )


def tracking_from_args(args: argparse.Namespace, root: Path) -> TrackingConfig:
    """Resolve where the run is recorded: flag, then environment, then a repo-local store.

    The local fallback is explicit rather than MLflow's own default, which drops a
    `mlflow.db` into whatever directory the command happens to run from.
    """
    if args.no_tracking:
        return TrackingConfig(enabled=False)
    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    artifact_location = None
    if not uri:
        store = root / "mlruns"
        uri = f"sqlite:///{store / 'mlflow.db'}"
        artifact_location = (store / "artifacts").as_uri()
    return TrackingConfig(
        enabled=True,
        tracking_uri=uri,
        experiment_name=args.experiment,
        artifact_location=artifact_location,
        registered_model_name=args.register_model,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_pipeline(config_from_args(args), verbose=0 if args.quiet else 1)
    except MlppError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"artifacts -> {result.session_dir} ({len(result.rows)} evaluations)")
    if result.run_id is not None:
        print(f"mlflow run -> {result.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

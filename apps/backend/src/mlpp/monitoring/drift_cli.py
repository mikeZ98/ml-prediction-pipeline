"""`mlpp-drift` — compare captured inference payloads with the training data.

Exit status is the verdict, so a scheduler or CI job can gate on it:

    0  no dataset drift
    2  dataset drift detected
    1  the comparison could not run (bad paths, too few rows, missing columns)

The column contract comes from the session's manifest; the reference defaults to
every CSV in the repo's TRAIN/ directory — the data the session was fitted on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

import pandas as pd

from mlpp.data import discover_csvs, read_csv_auto, read_table_auto
from mlpp.errors import DatasetError, MlppError
from mlpp.monitoring.drift import (
    DEFAULT_DRIFT_SHARE,
    DEFAULT_MIN_ROWS,
    DriftResult,
    detect_drift,
)
from mlpp.monitoring.payloads import read_payloads
from mlpp.session import read_manifest

DEFAULT_ROOT: Final = Path(__file__).resolve().parents[5]

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_DRIFT: Final = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlpp-drift",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session", type=Path, required=True, help="session the traffic hit")
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="captured payloads: a PAYLOADS/ directory, a .jsonl file, or a CSV/Parquet batch",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_ROOT / "TRAIN",
        help="training data: a directory of CSVs or one CSV/Parquet file (default: TRAIN/)",
    )
    parser.add_argument("--html", type=Path, help="write Evidently's interactive report here")
    parser.add_argument("--json", type=Path, help="write the machine-readable summary here")
    parser.add_argument(
        "--drift-share",
        type=float,
        default=DEFAULT_DRIFT_SHARE,
        help="share of drifted columns that counts as dataset drift (default: %(default)s)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help="refuse to test samples smaller than this (default: %(default)s)",
    )
    return parser


def load_reference(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = discover_csvs(path)
        if not files:
            raise DatasetError(f"no CSV files in {path}")
        return pd.concat([read_csv_auto(p) for p in files], ignore_index=True)
    return read_table_auto(path)


def load_current(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(f"{path} does not exist")
    if path.is_dir() or path.suffix.lower() == ".jsonl":
        return read_payloads(path)
    return read_table_auto(path)


def render(result: DriftResult) -> str:
    verdict = "DRIFT DETECTED" if result.dataset_drift else "no dataset drift"
    lines = [
        f"{verdict}: {result.drifted_columns}/{len(result.columns)} column(s) drifted "
        f"(share {result.drifted_share:.2f}, threshold {result.drift_share_threshold:.2f}; "
        f"reference {result.n_reference} rows, current {result.n_current} rows)",
    ]
    width = max(len(c.column) for c in result.columns)
    for column in result.columns:
        flag = "DRIFT" if column.drifted else "ok"
        lines.append(
            f"  {column.column:<{width}}  {flag:<5}  {column.method}: "
            f"{column.score:.4g} (threshold {column.threshold:g})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        contract = read_manifest(args.session).features
        result = detect_drift(
            load_reference(args.reference),
            load_current(args.current),
            contract,
            drift_share=args.drift_share,
            min_rows=args.min_rows,
            html_path=args.html,
        )
    except (MlppError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(render(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    if args.html is not None:
        print(f"report -> {args.html}")
    return EXIT_DRIFT if result.dataset_drift else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

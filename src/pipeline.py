from __future__ import annotations

import argparse
import logging

from .build_base_dataset import build_base_dataset
from .clean_data import clean_all
from .feature_engineering import build_feature_engineering
from .crawler import run_crawler
from .utils import setup_logging


logger = logging.getLogger(__name__)


def run_all(dry_run: bool = False) -> None:
    run_crawler(dry_run=dry_run)
    clean_all(dry_run=dry_run)
    build_base_dataset(dry_run=dry_run)   # session-level base
    build_feature_engineering(dry_run=dry_run)  # lap-level master


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 data-layer pipeline runner")
    parser.add_argument(
        "step",
        choices=["fetch", "clean", "build-base", "feature-engineering", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving any data to disk",
    )
    args = parser.parse_args()

    setup_logging()

    if args.step == "fetch":
        run_crawler(dry_run=args.dry_run)
    elif args.step == "clean":
        clean_all(dry_run=args.dry_run)
    elif args.step == "build-base":
        build_base_dataset(dry_run=args.dry_run)
    elif args.step == "feature-engineering":
        build_feature_engineering(dry_run=args.dry_run)
    elif args.step == "all":
        run_all(dry_run=args.dry_run)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()

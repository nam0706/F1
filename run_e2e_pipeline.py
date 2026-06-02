from __future__ import annotations

import logging

from src.clean_data import clean_all
from src.config import load_config
from src.crawler import run_crawler
from src.gold.build_gold import build_gold
from src.model.training import train_gold_finish_bucket_model
from src.utils import setup_logging
from src.validate_data import validate_cleaned_data

logger = logging.getLogger("E2E_Pipeline")

ALL_STEPS = ["crawl", "clean", "validate", "feature", "model"]


def _resolve_steps(config_steps: list[str], steps: list[str] | None, start_from: str | None) -> list[str]:
    if steps:
        unknown = sorted(set(steps) - set(ALL_STEPS))
        if unknown:
            raise ValueError(f"Unknown pipeline step(s): {unknown}. Valid steps: {ALL_STEPS}")
        return [step for step in ALL_STEPS if step in steps]

    if start_from:
        if start_from not in ALL_STEPS:
            raise ValueError(f"Unknown start_from={start_from!r}. Valid steps: {ALL_STEPS}")
        return ALL_STEPS[ALL_STEPS.index(start_from):]

    return [step for step in ALL_STEPS if step in config_steps]


def run_e2e_pipeline(
    steps: list[str] | None = None,
    start_from: str | None = None,
    dry_run: bool | None = None,
) -> None:
    """Run the configured F1 medallion pipeline."""
    config = load_config()
    setup_logging(
        level=config.log_level,
        log_file_path=config.log_file_path,
        log_to_file=config.log_to_file,
        retention_days=config.log_retention_days,
    )
    dry_run = config.dry_run_default if dry_run is None else dry_run
    to_run = _resolve_steps(config.steps_to_run, steps, start_from)

    print("\n" + "=" * 72)
    print("F1 DATA PIPELINE")
    print(f"Mode: {config.execution_mode}")
    print(f"Years: {config.start_year} -> {config.end_year}")
    print(f"Steps: {' -> '.join(to_run) if to_run else '(none)'}")
    if dry_run:
        print("Mode: dry run")
    print("=" * 72)

    if "crawl" in to_run:
        print("\n[1/5] Crawl Bronze raw data")
        crawl_meta = run_crawler(dry_run=dry_run)
        print(f"  Crawl status: {crawl_meta.get('status', 'unknown')}")
        print(f"  Sessions processed: {crawl_meta.get('total_sessions', 0):,}")
    else:
        print("\n[1/5] Crawl Bronze raw data - skipped")

    if "clean" in to_run:
        print("\n[2/5] Clean Bronze -> Silver")
        clean_all(dry_run=dry_run)
    else:
        print("\n[2/5] Clean Bronze -> Silver - skipped")

    if "validate" in to_run:
        print("\n[3/5] Validate Silver tables")
        validation_report = validate_cleaned_data()
        error_count = 0 if validation_report.empty else int((validation_report["severity"] == "error").sum())
        print(f"  Validation issues: {len(validation_report):,} ({error_count:,} errors)")
    else:
        print("\n[3/5] Validate Silver tables - skipped")

    if "feature" in to_run:
        print("\n[4/5] Build Gold datasets")
        gold_meta = build_gold(dry_run=dry_run)
        feature_meta = gold_meta.get("features") or {}
        print(f"  Gold status: {gold_meta.get('status', 'unknown')}")
        print(f"  Master features: {feature_meta.get('rows', 0):,} rows x {len(feature_meta.get('columns', []))} columns")
    else:
        print("\n[4/5] Build Gold datasets - skipped")

    if "model" in to_run:
        print("\n[5/5] Train Gold contract model")
        model_meta = train_gold_finish_bucket_model(dry_run=dry_run)
        print(f"  Model status: {model_meta.get('status', 'unknown')}")
        metrics = model_meta.get("metrics", {})
        if metrics:
            print(f"  Holdout macro F1: {metrics.get('f1_macro', 0.0):.4f}")
    else:
        print("\n[5/5] Train Gold contract model - skipped")

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    run_e2e_pipeline()

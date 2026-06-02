from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import load_config
from .utils import write_csv, write_markdown


logger = logging.getLogger(__name__)


SILVER_PARQUET_ENDPOINTS = {
    "meetings",
    "sessions",
    "drivers",
    "laps",
    "weather",
    "stints",
    "starting_grid",
    "intervals",
    "position",
    "session_result",
    "overtakes",
    "pit",
    "race_control",
    "car_data",
    "location",
}

KEY_MAP = {
    "meetings.parquet": ["meeting_key"],
    "sessions.parquet": ["session_key"],
    "drivers.parquet": ["session_key", "driver_number"],
    "laps.parquet": ["session_key", "driver_number", "lap_number"],
    "weather.parquet": ["session_key", "date"],
    "stints.parquet": ["session_key", "driver_number", "stint_number"],
    "position.parquet": ["session_key", "driver_number", "date"],
    "starting_grid.parquet": ["session_key", "driver_number"],
    "session_result.parquet": ["session_key", "driver_number"],
    "overtakes.parquet": ["session_key", "date", "overtaking_driver_number", "overtaken_driver_number"],
    "pit.parquet": ["session_key", "driver_number", "lap_number", "date"],
    "race_control.parquet": ["session_key", "date", "category", "message"],
}

STRUCTURAL_NULL_ALLOWED = {
    "race_control.driver_number",
    "race_control.lap_number",
    "race_control.flag",
    "pit.pit_duration",
    "pit.stop_duration",
}


def required_files_from_schema_contract(config) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {}
    for endpoint, contract in config.schema_contract.items():
        columns = list(contract.get("required_columns", []))
        if not columns:
            continue
        filename = f"{endpoint}.parquet" if endpoint in SILVER_PARQUET_ENDPOINTS else f"{endpoint}.csv"
        required[filename] = columns
    return required


def check_required_columns(available_cols: list[str], filename: str, columns: list[str]) -> list[dict[str, Any]]:
    missing = [col for col in columns if col not in available_cols]
    if not missing:
        return []
    return [{
        "file": filename,
        "severity": "error",
        "check": "required_columns",
        "message": f"Missing required columns: {missing}",
    }]


def _hash_key_frame(frame: pd.DataFrame, key: list[str]) -> np.ndarray:
    key_frame = frame[key].astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype=np.uint64)


def _validate_frame_chunk(
    chunk: pd.DataFrame,
    duplicate_key: list[str],
    seen_hashes: set[int],
    null_counts: dict[str, int],
) -> tuple[int, int]:
    rows = len(chunk)
    duplicate_count = 0

    for col, count in chunk.isna().sum().items():
        null_counts[col] = null_counts.get(col, 0) + int(count)

    key_cols = [col for col in duplicate_key if col in chunk.columns]
    if key_cols:
        hashes = _hash_key_frame(chunk, key_cols)
        if len(hashes):
            hash_series = pd.Series(hashes)
            duplicate_within_chunk = hash_series.duplicated().to_numpy()
            duplicate_from_prior_chunks = np.fromiter(
                (int(value) in seen_hashes for value in hashes),
                dtype=bool,
                count=len(hashes),
            )
            duplicate_count += int((duplicate_within_chunk | duplicate_from_prior_chunks).sum())
            seen_hashes.update(map(int, np.unique(hashes)))

    return rows, duplicate_count


def _append_missing_issues(
    issues: list[dict[str, Any]],
    filename: str,
    required_cols: list[str],
    null_counts: dict[str, int],
    rows: int,
    max_null_pct: float,
) -> None:
    if rows == 0:
        return

    endpoint = filename.replace(".parquet", "").replace(".csv", "")
    for col, count in null_counts.items():
        if count <= 0:
            continue
        pct = count / rows * 100
        qualified_name = f"{endpoint}.{col}"
        if qualified_name in STRUCTURAL_NULL_ALLOWED:
            severity = "info"
        elif col in required_cols and pct > max_null_pct:
            severity = "error"
        else:
            severity = "warning"
        issues.append({
            "file": filename,
            "severity": severity,
            "check": "missing_values",
            "message": f"Column {col} has {count} nulls ({pct:.1f}%)",
        })


def validate_parquet_chunked(
    path,
    filename: str,
    required_cols: list[str],
    duplicate_key: list[str],
    max_null_pct: float,
    max_duplicate_rate_pct: float,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    parquet_file = pq.ParquetFile(path)
    available_cols = parquet_file.schema.names
    issues.extend(check_required_columns(available_cols, filename, required_cols))

    selected_cols = sorted(set([col for col in required_cols + duplicate_key if col in available_cols]))
    if not selected_cols:
        return issues

    total_rows = 0
    duplicate_count = 0
    null_counts: dict[str, int] = {}
    seen_hashes: set[int] = set()

    for row_group_idx in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_group(row_group_idx, columns=selected_cols).to_pandas()
        rows, duplicates = _validate_frame_chunk(chunk, duplicate_key, seen_hashes, null_counts)
        total_rows += rows
        duplicate_count += duplicates

    duplicate_rate = duplicate_count / total_rows * 100 if total_rows else 0.0
    if duplicate_count and duplicate_rate > max_duplicate_rate_pct:
        issues.append({
            "file": filename,
            "severity": "warning",
            "check": "duplicate_key",
            "message": f"Found {duplicate_count} duplicate rows ({duplicate_rate:.4f}%) for key={duplicate_key}",
        })

    _append_missing_issues(issues, filename, required_cols, null_counts, total_rows, max_null_pct)
    logger.info("Validated %s with %s row group(s), %s rows", filename, parquet_file.num_row_groups, total_rows)
    return issues


def validate_csv_chunked(
    path,
    filename: str,
    required_cols: list[str],
    duplicate_key: list[str],
    max_null_pct: float,
    max_duplicate_rate_pct: float,
    chunk_size: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    header = pd.read_csv(path, nrows=0)
    issues.extend(check_required_columns(list(header.columns), filename, required_cols))

    total_rows = 0
    duplicate_count = 0
    null_counts: dict[str, int] = {}
    seen_hashes: set[int] = set()

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        rows, duplicates = _validate_frame_chunk(chunk, duplicate_key, seen_hashes, null_counts)
        total_rows += rows
        duplicate_count += duplicates

    duplicate_rate = duplicate_count / total_rows * 100 if total_rows else 0.0
    if duplicate_count and duplicate_rate > max_duplicate_rate_pct:
        issues.append({
            "file": filename,
            "severity": "warning",
            "check": "duplicate_key",
            "message": f"Found {duplicate_count} duplicate rows ({duplicate_rate:.4f}%) for key={duplicate_key}",
        })

    _append_missing_issues(issues, filename, required_cols, null_counts, total_rows, max_null_pct)
    return issues


def validate_cleaned_data(config_path: str | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    issues: list[dict[str, Any]] = []
    required_files = required_files_from_schema_contract(config)
    max_null_pct = float(config.quality_config.get("max_null_pct_for_required_columns", 0.0))
    max_duplicate_rate_pct = float(config.quality_config.get("max_duplicate_rate_pct", 0.0))
    chunk_size = int(config.quality_config.get("validation_chunk_size", 200_000))

    for filename, required_cols in required_files.items():
        path = config.cleaned_dir / filename
        if not path.exists():
            issues.append({
                "file": filename,
                "severity": "error",
                "check": "file_exists",
                "message": "File missing",
            })
            continue

        duplicate_key = KEY_MAP.get(filename, required_cols)
        if path.suffix == ".parquet":
            issues.extend(validate_parquet_chunked(
                path,
                filename,
                required_cols,
                duplicate_key,
                max_null_pct,
                max_duplicate_rate_pct,
            ))
        else:
            issues.extend(validate_csv_chunked(
                path,
                filename,
                required_cols,
                duplicate_key,
                max_null_pct,
                max_duplicate_rate_pct,
                chunk_size,
            ))

    report_df = pd.DataFrame(issues, columns=["file", "severity", "check", "message"])
    write_csv(report_df, config.reports_dir / "validation_report.csv")

    if report_df.empty:
        markdown = "# Validation Report\n\nNo validation issues found.\n"
    else:
        markdown = "# Validation Report\n\n" + report_df.to_markdown(index=False) + "\n"
    write_markdown(markdown, config.reports_dir / "validation_report.md")

    error_count = 0 if report_df.empty else int((report_df["severity"] == "error").sum())
    logger.info("Validation complete with %s issue(s), %s error(s)", len(report_df), error_count)
    return report_df


if __name__ == "__main__":
    validate_cleaned_data()

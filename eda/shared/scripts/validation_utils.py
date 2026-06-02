"""Streaming validation utilities for Bronze notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from file_utils import endpoint_files, iter_csv_endpoint, iter_parquet_endpoint, read_header


def endpoint_columns(raw_dir: Path, endpoint: str) -> list[str]:
    columns: list[str] = []
    for path in endpoint_files(raw_dir, endpoint):
        try:
            for column in read_header(path):
                if column not in columns:
                    columns.append(column)
        except Exception:
            continue
    return columns


def null_profile(
    raw_dir: Path,
    endpoint: str,
    is_telemetry: bool,
    chunksize: int = 200_000,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = expected_columns or None
    iterator = (
        iter_parquet_endpoint(raw_dir, endpoint, columns=columns)
        if is_telemetry
        else iter_csv_endpoint(raw_dir, endpoint, columns=columns, chunksize=chunksize)
    )
    rows = 0
    null_counts: dict[str, int] = {column: 0 for column in (expected_columns or [])}
    for chunk in iterator:
        base = chunk.drop(columns=["_source_file"], errors="ignore")
        rows += len(base)
        for column, count in base.isna().sum().items():
            null_counts[column] = null_counts.get(column, 0) + int(count)
    return pd.DataFrame([
        {"endpoint": endpoint, "column": column, "rows": rows, "null_count": count, "null_pct": count / rows * 100 if rows else 0}
        for column, count in null_counts.items()
    ])


def duplicate_count(raw_dir: Path, endpoint: str, key: list[str], chunksize: int = 200_000) -> tuple[int, int]:
    if not key:
        return 0, 0
    rows = 0
    duplicates = 0
    seen: set[int] = set()
    for chunk in iter_csv_endpoint(raw_dir, endpoint, columns=key, chunksize=chunksize):
        if not set(key).issubset(chunk.columns):
            continue
        base = chunk[key].astype("string").fillna("<NA>")
        hashes = pd.util.hash_pandas_object(base, index=False).to_numpy(dtype=np.uint64)
        rows += len(hashes)
        if len(hashes) == 0:
            continue
        hash_series = pd.Series(hashes)
        within = hash_series.duplicated().to_numpy()
        prior = np.fromiter((int(h) in seen for h in hashes), dtype=bool, count=len(hashes))
        duplicates += int((within | prior).sum())
        seen.update(map(int, np.unique(hashes)))
    return rows, duplicates


def range_violations(raw_dir: Path, endpoint: str, rules: dict[str, dict[str, Any]], is_telemetry: bool, chunksize: int = 200_000) -> pd.DataFrame:
    columns = list(rules)
    iterator = iter_parquet_endpoint(raw_dir, endpoint, columns=columns) if is_telemetry else iter_csv_endpoint(raw_dir, endpoint, columns=columns, chunksize=chunksize)
    records: dict[str, dict[str, Any]] = {}
    for chunk in iterator:
        for column, rule in rules.items():
            if column not in chunk.columns:
                continue
            values = pd.to_numeric(chunk[column], errors="coerce")
            present = values.notna()
            invalid = pd.Series(False, index=values.index)
            if not rule.get("allow_null", True):
                invalid |= values.isna()
            if "min" in rule:
                invalid |= present & (values < rule["min"])
            if "max" in rule:
                invalid |= present & (values > rule["max"])
            row = records.setdefault(column, {"endpoint": endpoint, "column": column, "checked_rows": 0, "violations": 0})
            row["checked_rows"] += int(len(values))
            invalid_values = values[invalid].dropna()
            row["violations"] += int(invalid.sum())
            if len(invalid_values):
                current_min = float(invalid_values.min())
                current_max = float(invalid_values.max())
                row["min_violation"] = current_min if "min_violation" not in row else min(row["min_violation"], current_min)
                row["max_violation"] = current_max if "max_violation" not in row else max(row["max_violation"], current_max)
                samples = row.setdefault("sample_values", [])
                if len(samples) < 10:
                    samples.extend(invalid_values.head(10 - len(samples)).tolist())
    return pd.DataFrame(records.values())

"""Shared file utilities for partitioned Bronze raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


ALLOWED_EXTENSIONS = {".csv", ".parquet", ".json"}


def endpoint_files(raw_dir: Path, endpoint: str) -> list[Path]:
    endpoint_dir = raw_dir / endpoint
    if not endpoint_dir.exists():
        return []
    return sorted(
        path
        for path in endpoint_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def infer_year(path: Path) -> int | None:
    for part in path.parts:
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def read_header(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path).schema.names
    return ["json_payload"]


def row_count(path: Path) -> int:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    if path.suffix.lower() == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    return 1


def inspect_file(path: Path, raw_dir: Path, endpoint: str) -> dict[str, Any]:
    columns: list[str] = []
    rows = 0
    corrupt = False
    error = ""
    try:
        columns = read_header(path)
        rows = row_count(path)
    except Exception as exc:  # pragma: no cover - diagnostic notebook path
        corrupt = True
        error = type(exc).__name__

    return {
        "endpoint": endpoint,
        "year": infer_year(path),
        "relative_path": str(path.relative_to(raw_dir)).replace("\\", "/"),
        "extension": path.suffix.lower(),
        "size_mb": path.stat().st_size / 1_000_000,
        "row_count": rows,
        "column_count": len(columns),
        "columns": "|".join(columns),
        "zero_rows": rows == 0,
        "corrupt": corrupt,
        "error": error,
    }


def build_file_inventory(raw_dir: Path, endpoints: Iterable[str]) -> pd.DataFrame:
    records = []
    for endpoint in endpoints:
        files = endpoint_files(raw_dir, endpoint)
        if not files:
            records.append({
                "endpoint": endpoint,
                "year": None,
                "relative_path": "",
                "extension": "",
                "size_mb": 0.0,
                "row_count": 0,
                "column_count": 0,
                "columns": "",
                "zero_rows": True,
                "corrupt": False,
                "error": "endpoint_missing",
            })
            continue
        for path in files:
            records.append(inspect_file(path, raw_dir, endpoint))
    return pd.DataFrame(records)


def iter_csv_endpoint(raw_dir: Path, endpoint: str, columns: list[str] | None = None, chunksize: int = 200_000):
    for path in endpoint_files(raw_dir, endpoint):
        if path.suffix.lower() != ".csv":
            continue
        try:
            available = pd.read_csv(path, nrows=0).columns.tolist()
            usecols = [col for col in (columns or available) if col in available]
            if columns is not None and not usecols:
                continue
            for chunk in pd.read_csv(path, usecols=usecols if columns is not None else None, chunksize=chunksize, low_memory=False):
                chunk["_source_file"] = str(path.relative_to(raw_dir)).replace("\\", "/")
                yield chunk
        except Exception:
            continue


def iter_parquet_endpoint(raw_dir: Path, endpoint: str, columns: list[str] | None = None):
    for path in endpoint_files(raw_dir, endpoint):
        if path.suffix.lower() != ".parquet":
            continue
        parquet_file = pq.ParquetFile(path)
        available = parquet_file.schema.names
        selected = [col for col in (columns or available) if col in available]
        if columns is not None and not selected:
            continue
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=selected if columns is not None else None)
            chunk = table.to_pandas()
            chunk["_source_file"] = str(path.relative_to(raw_dir)).replace("\\", "/")
            yield chunk

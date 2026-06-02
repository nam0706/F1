from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd


def build_interval_lap_features(base_laps: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Build gap and interval features at lap grain."""
    required_base = {"session_key", "driver_number", "lap_number", "lap_start_time"}
    required_intervals = {"session_key", "driver_number", "date"}
    missing_base = sorted(required_base - set(base_laps.columns))
    missing_intervals = sorted(required_intervals - set(intervals.columns))
    if missing_base:
        raise ValueError(f"base_laps is missing required columns: {missing_base}")
    if missing_intervals:
        raise ValueError(f"intervals is missing required columns: {missing_intervals}")

    base = base_laps[["session_key", "driver_number", "lap_number", "lap_start_time"]].copy()
    for col in ["session_key", "driver_number", "lap_number"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base["lap_start_time"] = pd.to_datetime(base["lap_start_time"], errors="coerce", utc=True)
    base = base.dropna(subset=["session_key", "driver_number", "lap_number", "lap_start_time"]).copy()
    base["session_key"] = base["session_key"].astype("int64")
    base["driver_number"] = base["driver_number"].astype("int64")
    base["lap_number"] = base["lap_number"].astype("int64")

    frame = intervals.copy()
    for col in ["session_key", "driver_number"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["session_key", "driver_number", "date"]).copy()
    frame["session_key"] = frame["session_key"].astype("int64")
    frame["driver_number"] = frame["driver_number"].astype("int64")

    text_candidates = [col for col in ["interval", "gap_to_leader"] if col in frame.columns]
    for col in text_candidates:
        frame[col] = frame[col].astype(str).str.strip()

    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
        return pd.to_numeric(cleaned, errors="coerce")

    if "interval" in frame.columns:
        frame["interval_value_seconds"] = _coerce_numeric_series(frame["interval"])
    else:
        frame["interval_value_seconds"] = np.nan
    if "gap_to_leader" in frame.columns:
        frame["gap_to_leader_seconds"] = _coerce_numeric_series(frame["gap_to_leader"])
    else:
        frame["gap_to_leader_seconds"] = np.nan

    join_cols = ["session_key", "driver_number", "lap_number"]
    results = []
    for session_key, session_base in base.groupby("session_key", sort=False):
        session_frame = frame[frame["session_key"] == session_key]
        if session_frame.empty:
            results.append(session_base.assign(
                interval_value_seconds=np.nan,
                gap_to_leader_seconds=np.nan,
                interval_status="UNKNOWN",
                gap_status="UNKNOWN",
            ))
            continue
        merged = pd.merge_asof(
            session_base.sort_values("lap_start_time"),
            session_frame.sort_values("date"),
            left_on="lap_start_time",
            right_on="date",
            by=["session_key", "driver_number"],
            direction="backward",
            allow_exact_matches=False,
        )
        merged["interval_status"] = np.where(merged["interval_value_seconds"].notna(), "KNOWN", "UNKNOWN")
        merged["gap_status"] = np.where(merged["gap_to_leader_seconds"].notna(), "KNOWN", "UNKNOWN")
        results.append(merged[join_cols + ["interval_value_seconds", "gap_to_leader_seconds", "interval_status", "gap_status"]])

    features = pd.concat(results, ignore_index=True)
    features["prev_gap_to_leader_seconds"] = features.groupby(["session_key", "driver_number"])["gap_to_leader_seconds"].shift(1)
    features["prev_interval_value_seconds"] = features.groupby(["session_key", "driver_number"])["interval_value_seconds"].shift(1)
    keep = join_cols + [
        "interval_value_seconds",
        "gap_to_leader_seconds",
        "prev_interval_value_seconds",
        "prev_gap_to_leader_seconds",
        "interval_status",
        "gap_status",
    ]
    return features[keep].sort_values(join_cols).reset_index(drop=True)


def build_interval_feature_contract(features: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/features/interval_lap_features.parquet",
        "grain": ["session_key", "driver_number", "lap_number"],
        "rows": int(len(features)),
        "columns": list(features.columns),
        "availability_rule": "Uses the latest interval snapshot strictly before current lap start; prior-gap columns are shifted by one lap.",
        "model_allowed_columns": [
            "interval_value_seconds",
            "gap_to_leader_seconds",
            "prev_interval_value_seconds",
            "prev_gap_to_leader_seconds",
            "interval_status",
            "gap_status",
        ],
    }


def write_interval_feature_contract(contract: dict, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

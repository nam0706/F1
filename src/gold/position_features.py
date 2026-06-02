from __future__ import annotations

from datetime import datetime
import json

import pandas as pd


def build_position_lap_features(base_laps: pd.DataFrame, position: pd.DataFrame) -> pd.DataFrame:
    """Build position state features at lap grain."""
    required_base = {"session_key", "driver_number", "lap_number", "lap_start_time"}
    required_position = {"session_key", "driver_number", "date", "position"}
    missing_base = sorted(required_base - set(base_laps.columns))
    missing_position = sorted(required_position - set(position.columns))
    if missing_base:
        raise ValueError(f"base_laps is missing required columns: {missing_base}")
    if missing_position:
        raise ValueError(f"position is missing required columns: {missing_position}")

    laps = base_laps[["session_key", "driver_number", "lap_number", "lap_start_time"]].copy()
    laps["lap_start_time"] = pd.to_datetime(laps["lap_start_time"], errors="coerce", utc=True)
    pos = position[["session_key", "driver_number", "date", "position"]].copy()
    pos["date"] = pd.to_datetime(pos["date"], errors="coerce", utc=True)
    pos["position"] = pd.to_numeric(pos["position"], errors="coerce")
    pos = pos.dropna(subset=["session_key", "driver_number", "date", "position"]).sort_values("date")

    parts = []
    for (session_key, driver_number), group in laps.groupby(["session_key", "driver_number"], sort=False):
        hist = pos[(pos["session_key"] == session_key) & (pos["driver_number"] == driver_number)][["date", "position"]]
        if hist.empty:
            block = group.copy()
            block["current_position"] = pd.NA
            parts.append(block)
            continue
        merged = pd.merge_asof(
            group.sort_values("lap_start_time"),
            hist.sort_values("date"),
            left_on="lap_start_time",
            right_on="date",
            direction="backward",
        )
        merged = merged.drop(columns=["date"], errors="ignore").rename(columns={"position": "current_position"})
        parts.append(merged)

    features = pd.concat(parts, ignore_index=True) if parts else laps.iloc[0:0].copy()
    features["current_position"] = pd.to_numeric(features["current_position"], errors="coerce")
    features["prev_position"] = features.groupby(["session_key", "driver_number"], sort=False)["current_position"].shift(1)
    features["position_delta_prev_lap"] = features["prev_position"] - features["current_position"]
    features["position_change_direction"] = features["position_delta_prev_lap"].fillna(0)
    features["is_leading"] = features["current_position"].eq(1).astype("int8")
    features["is_top3_live"] = features["current_position"].between(1, 3, inclusive="both").fillna(False).astype("int8")
    features["is_top10_live"] = features["current_position"].between(1, 10, inclusive="both").fillna(False).astype("int8")

    keep_cols = [
        "session_key",
        "driver_number",
        "lap_number",
        "current_position",
        "prev_position",
        "position_delta_prev_lap",
        "position_change_direction",
        "is_leading",
        "is_top3_live",
        "is_top10_live",
    ]
    return features[keep_cols].sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)


def build_position_feature_contract(features: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/features/position_lap_features.parquet",
        "grain": ["session_key", "driver_number", "lap_number"],
        "rows": int(len(features)),
        "columns": list(features.columns),
        "availability_rule": "Uses latest known position event at or before lap_start_time via merge_asof backward.",
        "model_allowed_columns": [
            "current_position",
            "prev_position",
            "position_delta_prev_lap",
            "position_change_direction",
            "is_leading",
            "is_top3_live",
            "is_top10_live",
        ],
    }


def write_position_feature_contract(contract: dict, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

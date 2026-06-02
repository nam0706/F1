from __future__ import annotations

from datetime import datetime
import json

import pandas as pd


def build_pit_lap_features(base_laps: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Build pit state features up to each prediction lap."""
    required_base = {"session_key", "driver_number", "lap_number"}
    required_pit = {"session_key", "driver_number", "lap_number"}
    missing_base = sorted(required_base - set(base_laps.columns))
    missing_pit = sorted(required_pit - set(pit.columns))
    if missing_base:
        raise ValueError(f"base_laps is missing required columns: {missing_base}")
    if missing_pit:
        raise ValueError(f"pit is missing required columns: {missing_pit}")

    features = base_laps[["session_key", "driver_number", "lap_number"]].copy()
    features = features.sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)

    pit_events = pit.copy()
    for col in ["session_key", "driver_number", "lap_number", "lane_duration", "stop_duration", "pit_duration"]:
        if col in pit_events.columns:
            pit_events[col] = pd.to_numeric(pit_events[col], errors="coerce")
    pit_events = pit_events.dropna(subset=["session_key", "driver_number", "lap_number"]).copy()
    pit_events["session_key"] = pit_events["session_key"].astype("int64")
    pit_events["driver_number"] = pit_events["driver_number"].astype("int64")
    pit_events["lap_number"] = pit_events["lap_number"].astype("int64")
    pit_events = pit_events.sort_values(["session_key", "driver_number", "lap_number"])
    pit_events = pit_events.drop_duplicates(["session_key", "driver_number", "lap_number"], keep="first")
    pit_events["pit_event"] = 1

    features = features.merge(
        pit_events[["session_key", "driver_number", "lap_number", "pit_event", "lane_duration", "stop_duration", "pit_duration"]],
        on=["session_key", "driver_number", "lap_number"],
        how="left",
    )
    features["pit_event"] = features["pit_event"].fillna(0).astype("int8")
    group_keys = ["session_key", "driver_number"]
    cumulative = features.groupby(group_keys, sort=False)["pit_event"].cumsum()
    features["pit_count_so_far"] = cumulative.groupby([features["session_key"], features["driver_number"]]).shift(1)
    features["pit_count_so_far"] = features["pit_count_so_far"].fillna(0).astype("int16")

    pit_lap_marker = features["lap_number"].where(features["pit_event"].eq(1))
    last_pit_lap = pit_lap_marker.groupby([features["session_key"], features["driver_number"]]).ffill()
    features["_last_pit_lap_before_current"] = last_pit_lap.groupby([features["session_key"], features["driver_number"]]).shift(1)
    features["laps_since_last_pit"] = features["lap_number"] - features["_last_pit_lap_before_current"]
    features.loc[features["_last_pit_lap_before_current"].isna(), "laps_since_last_pit"] = pd.NA

    for duration_col in ["lane_duration", "stop_duration", "pit_duration"]:
        if duration_col in features.columns:
            features[f"last_{duration_col}"] = (
                features[duration_col]
                .where(features["pit_event"].eq(1))
                .groupby([features["session_key"], features["driver_number"]]).ffill()
                .groupby([features["session_key"], features["driver_number"]]).shift(1)
            )

    keep_cols = [
        "session_key",
        "driver_number",
        "lap_number",
        "pit_count_so_far",
        "laps_since_last_pit",
        "last_lane_duration",
        "last_stop_duration",
        "last_pit_duration",
    ]
    keep_cols = [col for col in keep_cols if col in features.columns]
    return features[keep_cols].sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)


def build_pit_feature_contract(features: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/features/pit_lap_features.parquet",
        "grain": ["session_key", "driver_number", "lap_number"],
        "rows": int(len(features)),
        "columns": list(features.columns),
        "availability_rule": "Uses only pit events strictly before the current lap when computing cumulative state.",
        "model_allowed_columns": [
            "pit_count_so_far",
            "laps_since_last_pit",
            "last_lane_duration",
            "last_stop_duration",
            "last_pit_duration",
        ],
    }


def write_pit_feature_contract(contract: dict, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

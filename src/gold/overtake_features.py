from __future__ import annotations

from datetime import datetime
import json

import pandas as pd


def build_overtake_lap_features(base_laps: pd.DataFrame, overtakes: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative overtake features at lap grain."""
    required_base = {"session_key", "driver_number", "lap_number", "lap_start_time"}
    required_overtakes = {"session_key", "overtaken_driver_number", "date"}
    missing_base = sorted(required_base - set(base_laps.columns))
    missing_overtakes = sorted(required_overtakes - set(overtakes.columns))
    if missing_base:
        raise ValueError(f"base_laps is missing required columns: {missing_base}")
    if missing_overtakes:
        raise ValueError(f"overtakes is missing required columns: {missing_overtakes}")

    base = base_laps[["session_key", "driver_number", "lap_number", "lap_start_time"]].copy()
    for col in ["session_key", "driver_number", "lap_number"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base["lap_start_time"] = pd.to_datetime(base["lap_start_time"], errors="coerce", utc=True)
    base = base.dropna(subset=["session_key", "driver_number", "lap_number", "lap_start_time"]).copy()
    base["session_key"] = base["session_key"].astype("int64")
    base["driver_number"] = base["driver_number"].astype("int64")
    base["lap_number"] = base["lap_number"].astype("int64")

    frame = overtakes.copy()
    for col in ["session_key", "overtaken_driver_number", "overtaking_driver_number"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["session_key", "overtaken_driver_number", "date"]).copy()
    frame["session_key"] = frame["session_key"].astype("int64")
    frame["overtaken_driver_number"] = frame["overtaken_driver_number"].astype("int64")
    if "overtaking_driver_number" in frame.columns:
        frame["overtaking_driver_number"] = frame["overtaking_driver_number"].astype("Int64")

    lap_map = base.sort_values(["session_key", "driver_number", "lap_start_time"])

    def _assign_event_laps(events: pd.DataFrame) -> pd.DataFrame:
        assigned = []
        for (session_key, driver_number), event_group in events.groupby(["session_key", "driver_number"], sort=False):
            driver_laps = lap_map[
                (lap_map["session_key"].eq(session_key)) & (lap_map["driver_number"].eq(driver_number))
            ]
            if driver_laps.empty:
                continue
            assigned.append(
                pd.merge_asof(
                    event_group.sort_values("date"),
                    driver_laps[["lap_number", "lap_start_time"]].sort_values("lap_start_time"),
                    left_on="date",
                    right_on="lap_start_time",
                    direction="backward",
                )
            )
        return pd.concat(assigned, ignore_index=True) if assigned else pd.DataFrame()

    overtakes_for_events = frame.dropna(subset=["overtaking_driver_number"]).copy()
    overtakes_for_events["overtaking_driver_number"] = overtakes_for_events["overtaking_driver_number"].astype("int64")
    overtakes_for_events = overtakes_for_events.rename(
        columns={"overtaking_driver_number": "driver_number"}
    )
    overtakes_for = _assign_event_laps(overtakes_for_events)
    overtakes_for = (
        overtakes_for.dropna(subset=["lap_number"])
        .groupby(["session_key", "driver_number", "lap_number"])
        .size()
        .rename("overtakes_made_on_lap")
        .reset_index()
    )

    overtakes_against_events = frame.rename(columns={"overtaken_driver_number": "driver_number"})
    overtakes_against = _assign_event_laps(overtakes_against_events)
    overtakes_against = (
        overtakes_against.dropna(subset=["lap_number"])
        .groupby(["session_key", "driver_number", "lap_number"])
        .size()
        .rename("overtakes_lost_on_lap")
        .reset_index()
    )

    features = base[["session_key", "driver_number", "lap_number"]].merge(
        overtakes_for, on=["session_key", "driver_number", "lap_number"], how="left"
    )
    features = features.merge(overtakes_against, on=["session_key", "driver_number", "lap_number"], how="left")
    features["overtakes_made_on_lap"] = features["overtakes_made_on_lap"].fillna(0).astype("int64")
    features["overtakes_lost_on_lap"] = features["overtakes_lost_on_lap"].fillna(0).astype("int64")
    features = features.sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)
    grouped = features.groupby(["session_key", "driver_number"], sort=False)
    features["prev_lap_overtakes_made"] = grouped["overtakes_made_on_lap"].shift(1).fillna(0).astype("int64")
    features["prev_lap_overtakes_lost"] = grouped["overtakes_lost_on_lap"].shift(1).fillna(0).astype("int64")
    features["overtakes_made_so_far"] = grouped["overtakes_made_on_lap"].cumsum() - features["overtakes_made_on_lap"]
    features["overtakes_lost_so_far"] = grouped["overtakes_lost_on_lap"].cumsum() - features["overtakes_lost_on_lap"]
    features["net_overtakes_so_far"] = features["overtakes_made_so_far"] - features["overtakes_lost_so_far"]
    keep_cols = [
        "session_key",
        "driver_number",
        "lap_number",
        "prev_lap_overtakes_made",
        "prev_lap_overtakes_lost",
        "overtakes_made_so_far",
        "overtakes_lost_so_far",
        "net_overtakes_so_far",
    ]
    return features[keep_cols].sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)


def build_overtake_feature_contract(features: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/features/overtake_lap_features.parquet",
        "grain": ["session_key", "driver_number", "lap_number"],
        "rows": int(len(features)),
        "columns": list(features.columns),
        "availability_rule": "Assigns timestamped overtake events to laps, then shifts all event counts so lap N uses only events from laps before N.",
        "model_allowed_columns": [
            "prev_lap_overtakes_made",
            "prev_lap_overtakes_lost",
            "overtakes_made_so_far",
            "overtakes_lost_so_far",
            "net_overtakes_so_far",
        ],
    }


def write_overtake_feature_contract(contract: dict, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

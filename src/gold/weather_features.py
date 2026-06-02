from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd


def build_weather_lap_features(base_laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Build time-safe weather context features at lap grain."""
    required_base = {"session_key", "driver_number", "lap_number", "lap_start_time"}
    required_weather = {"session_key", "date"}
    missing_base = sorted(required_base - set(base_laps.columns))
    missing_weather = sorted(required_weather - set(weather.columns))
    if missing_base:
        raise ValueError(f"base_laps is missing required columns: {missing_base}")
    if missing_weather:
        raise ValueError(f"weather is missing required columns: {missing_weather}")

    base = base_laps[["session_key", "driver_number", "lap_number", "lap_start_time"]].copy()
    for col in ["session_key", "driver_number", "lap_number"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base["lap_start_time"] = pd.to_datetime(base["lap_start_time"], errors="coerce", utc=True)
    base = base.dropna(subset=["session_key", "driver_number", "lap_number", "lap_start_time"]).copy()
    base["session_key"] = base["session_key"].astype("int64")
    base["driver_number"] = base["driver_number"].astype("int64")
    base["lap_number"] = base["lap_number"].astype("int64")

    frame = weather.copy()
    frame["session_key"] = pd.to_numeric(frame["session_key"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    value_cols = [col for col in ["track_temperature", "air_temperature", "humidity", "pressure", "rainfall"] if col in frame.columns]
    for col in value_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["session_key", "date"]).copy()
    frame["session_key"] = frame["session_key"].astype("int64")
    frame = frame.sort_values(["session_key", "date"])

    join_cols = ["session_key", "driver_number", "lap_number"]
    results = []
    for session_key, session_base in base.groupby("session_key", sort=False):
        session_weather = frame[frame["session_key"] == session_key]
        if session_weather.empty:
            results.append(session_base.assign(
                track_temperature=np.nan,
                air_temperature=np.nan,
                humidity=np.nan,
                pressure=np.nan,
                rainfall=np.nan,
                is_wet_track=pd.NA,
            ))
            continue
        merged = pd.merge_asof(
            session_base.sort_values("lap_start_time"),
            session_weather,
            left_on="lap_start_time",
            right_on="date",
            by="session_key",
            direction="backward",
            allow_exact_matches=False,
        )
        results.append(merged)

    features = pd.concat(results, ignore_index=True)
    if "rainfall" in features.columns:
        features["is_wet_track"] = features["rainfall"].fillna(0).gt(0).astype("int8")
    else:
        features["is_wet_track"] = pd.NA
    features["track_temp_delta_prev_lap"] = features.groupby(["session_key", "driver_number"])["track_temperature"].diff()
    keep = join_cols + [
        "track_temperature",
        "air_temperature",
        "humidity",
        "pressure",
        "rainfall",
        "is_wet_track",
        "track_temp_delta_prev_lap",
    ]
    return features[keep].sort_values(join_cols).reset_index(drop=True)


def build_weather_feature_contract(features: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/features/weather_lap_features.parquet",
        "grain": ["session_key", "driver_number", "lap_number"],
        "rows": int(len(features)),
        "columns": list(features.columns),
        "availability_rule": "Uses the latest session weather observation strictly before the current lap start.",
        "model_allowed_columns": [
            "track_temperature",
            "air_temperature",
            "humidity",
            "pressure",
            "rainfall",
            "is_wet_track",
            "track_temp_delta_prev_lap",
        ],
    }


def write_weather_feature_contract(contract: dict, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

from __future__ import annotations

import pandas as pd


HORIZONS = {
    "horizon_25": 0.25,
    "horizon_50": 0.50,
    "horizon_75": 0.75,
}


def build_horizon_datasets(training_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build live_any_lap and fixed-horizon datasets."""
    required = {"session_key", "driver_number", "lap_number", "race_progress_pct", "target_finish_bucket"}
    missing = sorted(required - set(training_frame.columns))
    if missing:
        raise ValueError(f"training_frame is missing required columns: {missing}")

    frame = training_frame.copy()
    for col in ["session_key", "driver_number", "lap_number", "race_progress_pct"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["session_key", "driver_number", "lap_number", "race_progress_pct", "target_finish_bucket"]).copy()
    frame["session_key"] = frame["session_key"].astype("int64")
    frame["driver_number"] = frame["driver_number"].astype("int64")
    frame["lap_number"] = frame["lap_number"].astype("int64")
    frame = frame[frame["lap_number"].ge(2)].copy()
    frame = frame.sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)

    datasets: dict[str, pd.DataFrame] = {"live_any_lap": frame}
    horizon_frames = []
    keys = ["session_key", "driver_number"]
    for name, pct in HORIZONS.items():
        horizon = frame.copy()
        horizon["horizon"] = name
        horizon["horizon_pct"] = pct
        horizon["_distance_to_horizon"] = (horizon["race_progress_pct"] - pct).abs()
        horizon = horizon.sort_values(keys + ["_distance_to_horizon", "lap_number"])
        horizon = horizon.drop_duplicates(keys, keep="first").drop(columns=["_distance_to_horizon"])
        datasets[name] = horizon.reset_index(drop=True)
        horizon_frames.append(datasets[name])

    datasets["horizon_panel"] = pd.concat(horizon_frames, ignore_index=True) if horizon_frames else pd.DataFrame()
    return datasets

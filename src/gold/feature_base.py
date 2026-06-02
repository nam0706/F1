from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd

from src.gold.scheduled_laps import (
    GRAND_PRIX_SCHEDULED_LAPS,
    SPRINT_SCHEDULED_LAPS,
)


BASE_FEATURE_COLUMNS = [
    "meeting_key",
    "session_key",
    "driver_number",
    "lap_number",
    "lap_start_time",
    "event_type",
    "session_name",
    "year",
    "circuit_short_name",
    "country_name",
    "full_name",
    "name_acronym",
    "team_name",
    "country_code",
    "grid_position",
    "scheduled_laps",
    "scheduled_laps_source",
    "race_progress_pct",
    "laps_to_go",
    "is_pit_out_lap",
    "is_outlier_lap",
]

BASE_MODEL_ALLOWED_COLUMNS = [
    "event_type",
    "year",
    "circuit_short_name",
    "country_name",
    "team_name",
    "country_code",
    "grid_position",
    "scheduled_laps",
    "race_progress_pct",
    "laps_to_go",
    "is_pit_out_lap",
    "is_outlier_lap",
]


def _build_scheduled_lap_map(base: pd.DataFrame) -> tuple[pd.Series, dict]:
    if base.empty:
        return pd.Series(dtype="float64"), {
            "source": "unavailable",
            "mapped_sessions": 0,
            "fallback_sessions": 0,
        }

    sessions = base[["session_key", "session_name", "circuit_short_name"]].drop_duplicates("session_key").copy()
    observed_max = base.groupby("session_key")["lap_number"].max()

    mapped = {}
    for row in sessions.itertuples(index=False):
        if str(row.session_name).lower() == "sprint":
            scheduled = SPRINT_SCHEDULED_LAPS.get(row.circuit_short_name)
        else:
            scheduled = GRAND_PRIX_SCHEDULED_LAPS.get(row.circuit_short_name)
        mapped[row.session_key] = scheduled

    scheduled = pd.Series(mapped, dtype="float64")
    missing = scheduled.isna()
    if missing.any():
        scheduled.loc[missing] = scheduled.loc[missing].index.to_series().map(observed_max)

    return scheduled, {
        "source": "circuit/session_name scheduled-lap mapping with session-level observed max fallback",
        "mapped_sessions": int((~missing).sum()),
        "fallback_sessions": int(missing.sum()),
        "fallback_session_keys": [int(value) for value in scheduled.loc[missing].index.tolist()],
    }


def build_base_lap_frame(
    laps: pd.DataFrame,
    sessions: pd.DataFrame,
    drivers: pd.DataFrame,
    starting_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Build the canonical driver-session-lap base frame."""
    required = {"session_key", "driver_number", "lap_number"}
    missing = sorted(required - set(laps.columns))
    if missing:
        raise ValueError(f"laps is missing required columns: {missing}")

    base = laps.copy()
    for col in ["meeting_key", "session_key", "driver_number", "lap_number"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    base = base.dropna(subset=["session_key", "driver_number", "lap_number"]).copy()
    base["session_key"] = base["session_key"].astype("int64")
    base["driver_number"] = base["driver_number"].astype("int64")
    base["lap_number"] = base["lap_number"].astype("int64")
    if "meeting_key" in base.columns:
        base["meeting_key"] = pd.to_numeric(base["meeting_key"], errors="coerce").astype("Int64")

    if "date_start" in base.columns:
        base["lap_start_time"] = pd.to_datetime(base["date_start"], errors="coerce", utc=True)
        base = base.drop(columns=["date_start"])

    sessions_dim = sessions.copy()
    sessions_dim["event_type"] = np.where(
        sessions_dim["session_name"].astype(str).str.lower().eq("sprint"),
        "SPRINT_RACE",
        "GRAND_PRIX_RACE",
    )
    session_cols = [
        col for col in [
            "session_key", "meeting_key", "session_name", "event_type", "year",
            "circuit_short_name", "country_name", "date_start", "date_end",
        ]
        if col in sessions_dim.columns
    ]
    sessions_dim = sessions_dim[session_cols].drop_duplicates("session_key")
    sessions_dim = sessions_dim.rename(columns={"date_start": "session_start_time", "date_end": "session_end_time"})

    drivers_dim = drivers.copy()
    driver_cols = [col for col in ["session_key", "driver_number", "full_name", "name_acronym", "team_name", "country_code"] if col in drivers_dim.columns]
    drivers_dim = drivers_dim[driver_cols].drop_duplicates(["session_key", "driver_number"])

    grid = starting_grid.copy()
    if "position" in grid.columns:
        grid["grid_position"] = pd.to_numeric(grid["position"], errors="coerce")
    grid_cols = [col for col in ["session_key", "driver_number", "grid_position"] if col in grid.columns]
    grid = grid[grid_cols].drop_duplicates(["session_key", "driver_number"])

    base = base.merge(sessions_dim, on="session_key", how="left", suffixes=("", "_session"))
    base = base.merge(drivers_dim, on=["session_key", "driver_number"], how="left")
    base = base.merge(grid, on=["session_key", "driver_number"], how="left")

    scheduled_laps, scheduled_meta = _build_scheduled_lap_map(base)
    base["scheduled_laps"] = base["session_key"].map(scheduled_laps)
    base["scheduled_laps_source"] = np.where(
        base["session_key"].isin(scheduled_meta["fallback_session_keys"]),
        "observed_session_max_fallback",
        "circuit_schedule_map",
    )
    base["race_progress_pct"] = base["lap_number"] / base["scheduled_laps"]
    base["laps_to_go"] = (base["scheduled_laps"] - base["lap_number"]).clip(lower=0)

    for col in ["is_pit_out_lap", "is_outlier_lap"]:
        if col in base.columns:
            base[col] = base[col].fillna(False).astype("int8")

    keep_columns = [col for col in BASE_FEATURE_COLUMNS if col in base.columns]
    base = base[keep_columns].sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/features/master_lap_features.parquet",
        "grain": ["session_key", "driver_number", "lap_number"],
        "rows": int(len(base)),
        "columns": keep_columns,
        "availability_rule": "Uses only static session/driver context and current lap index state available before lap start.",
        "model_allowed_columns": [col for col in BASE_MODEL_ALLOWED_COLUMNS if col in keep_columns],
        "scheduled_lap_source": scheduled_meta["source"],
        "scheduled_lap_metadata": scheduled_meta,
    }
    return base, metadata


def build_master_lap_features(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join approved Gold feature tables into master_lap_features."""
    if "base_lap_frame" not in feature_tables:
        raise ValueError("feature_tables must include 'base_lap_frame'")
    return feature_tables["base_lap_frame"].copy()


def write_feature_contract(metadata: dict, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

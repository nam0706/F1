from __future__ import annotations

from datetime import datetime
import json

import pandas as pd


def build_stint_lap_features(base_laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Build compound, stint, and tyre-age features at lap grain."""
    required_base = {"session_key", "driver_number", "lap_number"}
    required_stints = {"session_key", "driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"}
    missing_base = sorted(required_base - set(base_laps.columns))
    missing_stints = sorted(required_stints - set(stints.columns))
    if missing_base:
        raise ValueError(f"base_laps is missing required columns: {missing_base}")
    if missing_stints:
        raise ValueError(f"stints is missing required columns: {missing_stints}")

    laps = base_laps[["session_key", "driver_number", "lap_number"]].copy()
    for col in ["session_key", "driver_number", "lap_number"]:
        laps[col] = pd.to_numeric(laps[col], errors="coerce")
    laps = laps.dropna(subset=["session_key", "driver_number", "lap_number"]).copy()
    laps["session_key"] = laps["session_key"].astype("int64")
    laps["driver_number"] = laps["driver_number"].astype("int64")
    laps["lap_number"] = laps["lap_number"].astype("int64")

    stint_dim = stints.copy()
    for col in ["session_key", "driver_number", "stint_number", "lap_start", "lap_end", "tyre_age_at_start"]:
        stint_dim[col] = pd.to_numeric(stint_dim[col], errors="coerce")
    stint_dim["compound"] = stint_dim["compound"].fillna("UNKNOWN").astype(str).str.upper()
    stint_dim = stint_dim.dropna(subset=["session_key", "driver_number", "stint_number", "lap_start", "lap_end"]).copy()
    stint_dim["session_key"] = stint_dim["session_key"].astype("int64")
    stint_dim["driver_number"] = stint_dim["driver_number"].astype("int64")
    stint_dim["stint_number"] = stint_dim["stint_number"].astype("int64")
    stint_dim["lap_start"] = stint_dim["lap_start"].astype("int64")
    stint_dim["lap_end"] = stint_dim["lap_end"].astype("int64")
    stint_dim["tyre_age_at_start"] = stint_dim["tyre_age_at_start"].fillna(0).astype("int64")

    merged = laps.merge(
        stint_dim,
        on=["session_key", "driver_number"],
        how="left",
    )
    matched = merged[(merged["lap_number"] >= merged["lap_start"]) & (merged["lap_number"] <= merged["lap_end"])].copy()
    matched_keys = matched[["session_key", "driver_number", "lap_number"]].drop_duplicates()
    unmatched = laps.merge(
        matched_keys.assign(_matched=1),
        on=["session_key", "driver_number", "lap_number"],
        how="left",
    )
    unmatched = unmatched[unmatched["_matched"].isna()].drop(columns=["_matched"])
    if not unmatched.empty:
        unmatched = unmatched.assign(
            stint_number=pd.NA,
            compound="UNKNOWN",
            tyre_age_at_start=pd.NA,
            lap_start=pd.NA,
            lap_end=pd.NA,
        )
        merged = pd.concat([matched, unmatched], ignore_index=True, sort=False)
    else:
        merged = matched
    merged = merged.sort_values(["session_key", "driver_number", "lap_number", "stint_number"])
    merged = merged.drop_duplicates(["session_key", "driver_number", "lap_number"], keep="last")

    merged["stint_lap_index"] = merged["lap_number"] - merged["lap_start"] + 1
    merged["current_tyre_age"] = (merged["stint_lap_index"] - 1 + merged["tyre_age_at_start"]).clip(lower=0)
    merged["laps_remaining_in_stint"] = (merged["lap_end"] - merged["lap_number"]).clip(lower=0)
    merged["is_first_stint"] = merged["stint_number"].eq(1).fillna(False).astype("int8")

    keep_cols = [
        "session_key",
        "driver_number",
        "lap_number",
        "stint_number",
        "compound",
        "tyre_age_at_start",
        "stint_lap_index",
        "current_tyre_age",
        "laps_remaining_in_stint",
        "is_first_stint",
    ]
    return merged[keep_cols].sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)


def build_stint_feature_contract(features: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/features/stint_lap_features.parquet",
        "grain": ["session_key", "driver_number", "lap_number"],
        "rows": int(len(features)),
        "columns": list(features.columns),
        "availability_rule": "Uses current stint assignment for the current lap with no future-lap rolling information.",
        "model_allowed_columns": [
            "stint_number",
            "compound",
            "tyre_age_at_start",
            "stint_lap_index",
            "current_tyre_age",
            "laps_remaining_in_stint",
            "is_first_stint",
        ],
    }


def write_stint_feature_contract(contract: dict, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

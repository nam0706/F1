from __future__ import annotations

import logging

import pandas as pd

from .config import load_config
from .utils import read_csv, setup_logging, write_csv, write_markdown


logger = logging.getLogger(__name__)


def _read_optional_csv(path):
    return read_csv(path) if path.exists() else pd.DataFrame()


def summarize_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty or "session_key" not in weather_df.columns:
        return pd.DataFrame()
    agg_cols = [
        "air_temperature",
        "track_temperature",
        "humidity",
        "pressure",
        "wind_speed",
    ]
    existing = [col for col in agg_cols if col in weather_df.columns]
    if not existing:
        return pd.DataFrame()
    result = weather_df.groupby("session_key", as_index=False)[existing].mean()
    result = result.rename(columns={col: f"{col}_mean" for col in existing})
    if "rainfall" in weather_df.columns:
        rainfall = weather_df.groupby("session_key", as_index=False)["rainfall"].max()
        rainfall = rainfall.rename(columns={"rainfall": "rainfall_max"})
        result = result.merge(rainfall, on="session_key", how="left")
    return result


def summarize_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    required = {"session_key", "driver_number", "lap_duration"}
    if laps_df.empty or not required.issubset(laps_df.columns):
        return pd.DataFrame()
    grouped = laps_df.groupby(["session_key", "driver_number"])["lap_duration"]
    return grouped.agg(
        lap_count="count",
        avg_lap_duration="mean",
        best_lap_duration="min",
        median_lap_duration="median",
        lap_duration_std="std",
    ).reset_index()


def summarize_stints(stints_df: pd.DataFrame) -> pd.DataFrame:
    required = {"session_key", "driver_number"}
    if stints_df.empty or not required.issubset(stints_df.columns):
        return pd.DataFrame()

    aggregations = {"stint_number": "nunique"} if "stint_number" in stints_df.columns else {}
    if "tyre_age_at_start" in stints_df.columns:
        aggregations["tyre_age_at_start"] = "max"

    result = stints_df.groupby(["session_key", "driver_number"], as_index=False).agg(aggregations)
    result = result.rename(columns={"stint_number": "stint_count", "tyre_age_at_start": "max_tyre_age_at_start"})

    if "compound" in stints_df.columns:
        compounds = stints_df.sort_values(["session_key", "driver_number"])
        first_compound = compounds.groupby(["session_key", "driver_number"], as_index=False)["compound"].first()
        last_compound = compounds.groupby(["session_key", "driver_number"], as_index=False)["compound"].last()
        compound_count = compounds.groupby(["session_key", "driver_number"], as_index=False)["compound"].nunique()
        first_compound = first_compound.rename(columns={"compound": "first_compound"})
        last_compound = last_compound.rename(columns={"compound": "last_compound"})
        compound_count = compound_count.rename(columns={"compound": "compound_count"})
        result = result.merge(first_compound, on=["session_key", "driver_number"], how="left")
        result = result.merge(last_compound, on=["session_key", "driver_number"], how="left")
        result = result.merge(compound_count, on=["session_key", "driver_number"], how="left")

    return result


def build_base_dataset(config_path: str | None = None, dry_run: bool = False) -> pd.DataFrame:
    config = load_config(config_path)
    if dry_run:
        print("\n--- [Dry Run] Building base dataset (No files will be saved) ---")

    sessions = _read_optional_csv(config.cleaned_dir / "sessions.csv")
    drivers = _read_optional_csv(config.cleaned_dir / "drivers.csv")
    grid = _read_optional_csv(config.cleaned_dir / "starting_grid.csv")
    results = _read_optional_csv(config.cleaned_dir / "session_results.csv")
    weather = _read_optional_csv(config.cleaned_dir / "weather.csv")
    laps = _read_optional_csv(config.cleaned_dir / "laps.csv")
    stints = _read_optional_csv(config.cleaned_dir / "stints.csv")

    if drivers.empty:
        raise RuntimeError("Cannot build base dataset because cleaned drivers.csv is missing or empty")

    base = drivers.copy()

    session_cols = [
        "session_key",
        "year",
        "session_type",
        "session_name",
        "circuit_key",
        "circuit_short_name",
        "location",
        "country_name",
        "date_start",
        "date_end",
    ]
    if not sessions.empty:
        keep = [col for col in session_cols if col in sessions.columns]
        base = base.merge(sessions[keep].drop_duplicates("session_key"), on="session_key", how="left")

    if not grid.empty and {"session_key", "driver_number", "position"}.issubset(grid.columns):
        base = base.merge(
            grid[["session_key", "driver_number", "position"]].rename(columns={"position": "grid_position"}),
            on=["session_key", "driver_number"],
            how="left",
        )

    if not results.empty and {"session_key", "driver_number", "position"}.issubset(results.columns):
        result_cols = ["session_key", "driver_number", "position"]
        if "status" in results.columns:
            result_cols.append("status")
        base = base.merge(
            results[result_cols].rename(columns={"position": "final_position"}),
            on=["session_key", "driver_number"],
            how="left",
        )

    weather_summary = summarize_weather(weather)
    if not weather_summary.empty:
        base = base.merge(weather_summary, on="session_key", how="left")

    lap_summary = summarize_laps(laps)
    if not lap_summary.empty:
        base = base.merge(lap_summary, on=["session_key", "driver_number"], how="left")

    stint_summary = summarize_stints(stints)
    if not stint_summary.empty:
        base = base.merge(stint_summary, on=["session_key", "driver_number"], how="left")

    if "final_position" in base.columns:
        final_position = pd.to_numeric(base["final_position"], errors="coerce")
        base["target_win"] = (final_position == 1).astype("Int64")
        base["target_podium"] = final_position.between(1, 3).astype("Int64")
        base["target_top10"] = final_position.between(1, 10).astype("Int64")

    base = base.drop_duplicates(subset=["session_key", "driver_number"]).reset_index(drop=True)

    if dry_run:
        print(f"  [Dry Run] Final dataset size: {len(base)} rows, {len(base.columns)} columns")
        print("  [Dry Run] Would save to data/processed/driver_session_base.csv")
    else:
        write_csv(base, config.processed_dir / "driver_session_base.csv")
        try:
            base.to_parquet(config.processed_dir / "driver_session_base.parquet", index=False)
        except Exception as exc:
            logger.warning("Could not write parquet output: %s", exc)

    if not dry_run:
        markdown = "# Driver Session Base Dataset\n\n"
        markdown += f"Rows: {len(base):,}\n\n"
        markdown += f"Columns: {len(base.columns):,}\n\n"
        markdown += "Primary key: `session_key + driver_number`\n\n"
        markdown += "This dataset is the main handoff artifact for the ML team.\n"
        write_markdown(markdown, config.reports_dir / "base_dataset_report.md")

    logger.info("Built driver_session_base with %s rows and %s columns", len(base), len(base.columns))
    return base


if __name__ == "__main__":
    setup_logging()
    build_base_dataset()

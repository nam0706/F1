from pathlib import Path

from run_e2e_pipeline import run_e2e_pipeline
from src.feature_engineering import build_feature_engineering, model_feature_columns


LIVE_BEFORE_LAP_BLOCKED = {
    "final_position",
    "target_win",
    "target_podium",
    "target_top10",
    "pit_stop_count_current_lap",
    "pit_duration_sum_current_lap",
    "pit_duration_mean_current_lap",
    "stop_duration_mean_current_lap",
    "lane_duration_mean_current_lap",
    "is_pit_current_lap",
    "race_control_events_current_lap",
    "yellow_flag_events_current_lap",
    "green_flag_events_current_lap",
    "yellow_flag_active_current_lap",
}


def test_configured_pipeline_smoke():
    """Current YAML intentionally runs clean only; this should not require stale base artifacts."""
    run_e2e_pipeline(dry_run=True)


def test_feature_build_preserves_lap_grain_and_blocks_live_leakage():
    cleaned_laps = Path("data/cleaned/laps.csv")
    if not cleaned_laps.exists():
        print("SKIP: data/cleaned/laps.csv is missing; run clean first.")
        return

    master = build_feature_engineering(dry_run=True)
    key = ["session_key", "driver_number", "lap_number"]
    assert master.duplicated(key).sum() == 0

    safe_features = set(model_feature_columns(master))
    leaked = LIVE_BEFORE_LAP_BLOCKED & safe_features
    assert not leaked, f"Live-before-lap feature list includes leakage columns: {sorted(leaked)}"

    assert "pit_stop_count_before_lap" in master.columns
    assert "race_control_events_before_lap" in master.columns
    assert "loc_prev_loc_points_count" in master.columns or not Path("data/cleaned/location.parquet").exists()


if __name__ == "__main__":
    test_configured_pipeline_smoke()
    test_feature_build_preserves_lap_grain_and_blocks_live_leakage()

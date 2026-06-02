"""Shared configuration for EDA notebooks.

This module adapts the project-level pipeline configuration to the notebook
workflow described in the ETL plan. Raw data is endpoint/year partitioned, so
the "file" names below are logical endpoint artifacts rather than flat files
under data/raw.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402


PIPELINE_CONFIG = load_config()

RAW_DATA_PATH = PIPELINE_CONFIG.raw_dir
CLEANED_DATA_PATH = PIPELINE_CONFIG.cleaned_dir
FEATURES_DATA_PATH = PIPELINE_CONFIG.features_dir
REPORTS_PATH = PIPELINE_CONFIG.reports_dir

OUTPUT_PATHS = {
    "bronze_tables": PROJECT_ROOT / "eda" / "bronze" / "outputs" / "tables",
    "bronze_charts": PROJECT_ROOT / "eda" / "bronze" / "outputs" / "charts",
    "bronze_reports": PROJECT_ROOT / "eda" / "bronze" / "outputs" / "reports",
    "bronze_insights": PROJECT_ROOT / "eda" / "bronze" / "insights",
    "bronze_checkpoints": PROJECT_ROOT / "eda" / "bronze" / "checkpoints",
}

EXPECTED_ENDPOINTS = [
    "meetings",
    "sessions",
    "drivers",
    "session_result",
    "laps",
    "weather",
    "stints",
    "starting_grid",
    "intervals",
    "position",
    "overtakes",
    "pit",
    "race_control",
    "car_data",
    "location",
]

TELEMETRY_ENDPOINTS = {"car_data", "location"}
TELEMETRY_FILES = ["car_data.parquet", "location.parquet"]

CRITICAL_COLUMNS = {
    endpoint: list(contract.get("required_columns", []))
    for endpoint, contract in PIPELINE_CONFIG.schema_contract.items()
}

TECHNICAL_KEY_COLUMNS = {
    "meeting_key",
    "session_key",
    "driver_number",
    "lap_number",
    "stint_number",
    "date",
}

DOMAIN_REVIEW_COLUMNS = {
    "lap_duration",
    "position",
    "duration",
    "number_of_laps",
    "gap_to_leader",
}

STRUCTURAL_OPTIONAL_COLUMNS = {
    "points",
    "segments_sector_1",
    "segments_sector_2",
    "segments_sector_3",
    "i1_speed",
    "i2_speed",
    "st_speed",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "pit_duration",
}

ALLOWED_NULL_SCENARIOS = {
    "race_control.driver_number": "Race-control messages may apply to track/session conditions rather than a specific driver.",
    "race_control.lap_number": "Race-control messages may be emitted before a lap context is available.",
    "race_control.flag": "Only populated when a flag is explicitly shown.",
    "pit.pit_duration": "Only populated when a pit stop duration is available.",
    "drivers.country_code": "Can be missing for guest or special-case drivers.",
}

VALIDATION_SEVERITY = {
    "file_missing": "BLOCKER",
    "file_corrupt": "BLOCKER",
    "zero_row_file": "BLOCKER",
    "critical_column_missing": "BLOCKER",
    "pk_duplicate": "BLOCKER",
    "fk_orphan": "BLOCKER",
    "technical_key_null": "BLOCKER",
    "allowed_structural_null": "INFO",
    "domain_semantic_null": "WARNING",
    "optional_column_null": "INFO",
    "range_violation_physical": "WARNING",
    "range_violation_weather": "INFO",
    "range_violation_contextual": "WARNING",
    "lap_continuity_gap": "WARNING",
    "telemetry_time_gap": "WARNING",
    "extra_column": "INFO",
}

RANGE_SEVERITY_OVERRIDES = {
    "weather.pressure": "INFO",
    "laps.lap_duration": "WARNING",
    "race_control.driver_number": "INFO",
    "race_control.lap_number": "INFO",
}

PRIMARY_KEYS = {
    "meetings": ["meeting_key"],
    "sessions": ["session_key"],
    "drivers": ["session_key", "driver_number"],
    "laps": ["session_key", "driver_number", "lap_number"],
    "session_result": ["session_key", "driver_number"],
    "starting_grid": ["session_key", "driver_number"],
    "stints": ["session_key", "driver_number", "stint_number"],
    "weather": ["session_key", "date"],
    "overtakes": ["session_key", "date", "overtaking_driver_number", "overtaken_driver_number"],
    "position": ["session_key", "driver_number", "date"],
    "intervals": ["session_key", "driver_number", "date"],
    "pit": ["session_key", "driver_number", "date"],
    "race_control": ["session_key", "date", "category", "message"],
}

FOREIGN_KEYS = [
    {"child": "sessions", "child_key": "meeting_key", "parent": "meetings", "parent_key": "meeting_key"},
    {"child": "laps", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "session_result", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "starting_grid", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "weather", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "stints", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "intervals", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "position", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "overtakes", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "pit", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
    {"child": "race_control", "child_key": "session_key", "parent": "sessions", "parent_key": "session_key"},
]

RANGE_RULES = {
    "driver_number": {"min": 1, "max": 99, "allow_null": False},
    "position": {"min": 0, "max": 25, "allow_null": True},
    "points": {"min": 0, "max": 26, "allow_null": True},
    "lap_number": {"min": 1, "allow_null": False},
    "lap_duration": {"min": 0, "max": 600, "allow_null": True},
    "duration": {"min": 0, "allow_null": True},
    "speed": {"min": 0, "max": 380, "allow_null": True},
    "rpm": {"min": 0, "max": 15000, "allow_null": True},
    "gear": {"min": -1, "max": 8, "allow_null": True},
    "throttle": {"min": 0, "max": 100, "allow_null": True},
    "brake": {"min": 0, "max": 100, "allow_null": True},
    "track_temperature": {"min": -10, "max": 80, "allow_null": True},
    "air_temperature": {"min": -10, "max": 60, "allow_null": True},
    "humidity": {"min": 0, "max": 100, "allow_null": True},
    "pressure": {"min": 850, "max": 1100, "allow_null": True},
    "wind_speed": {"min": 0, "max": 150, "allow_null": True},
    "year": {"min": 2024, "max": 2026, "allow_null": False},
}

THRESHOLDS = {
    **PIPELINE_CONFIG.quality_config,
    "max_null_percentage_critical": 0,
    "max_null_percentage_warning": 30,
    "max_duplicate_percentage": 0.1,
}

TELEMETRY_CONFIG = {
    "batch_size_rows": int(PIPELINE_CONFIG.quality_config.get("validation_chunk_size", 200_000)),
    "validation_mode": PIPELINE_CONFIG.quality_config.get("telemetry_validation_mode", "row_group"),
}

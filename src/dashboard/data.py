from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.dashboard.paths import PATHS


OVERVIEW_COLUMNS = [
    "session_key",
    "driver_number",
    "lap_number",
    "lap_start_time",
    "event_type",
    "year",
    "circuit_short_name",
    "country_name",
    "full_name",
    "name_acronym",
    "team_name",
    "grid_position",
    "race_progress_pct",
    "laps_to_go",
    "current_position",
    "compound",
    "current_tyre_age",
    "pit_count_so_far",
    "laps_since_last_pit",
    "track_temperature",
    "gap_to_leader_seconds",
    "target_finish_bucket",
    "target_finish_bucket_label",
]

PREDICTOR_CONTEXT_COLUMNS = [
    "session_key",
    "driver_number",
    "lap_number",
    "lap_start_time",
    "event_type",
    "year",
    "circuit_short_name",
    "country_name",
    "full_name",
    "name_acronym",
    "team_name",
    "current_position",
    "compound",
    "current_tyre_age",
    "pit_count_so_far",
    "laps_since_last_pit",
    "track_temperature",
    "gap_to_leader_seconds",
    "target_finish_bucket",
    "target_finish_bucket_label",
]


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading Gold training data...")
def load_training_data() -> pd.DataFrame:
    return pd.read_parquet(PATHS.training_path, columns=OVERVIEW_COLUMNS)


@st.cache_data(show_spinner="Loading model frame...")
def load_model_frame() -> pd.DataFrame:
    metadata = read_json(PATHS.model_metadata_path)
    columns = list(dict.fromkeys(PREDICTOR_CONTEXT_COLUMNS + metadata["feature_columns"]))
    frame = pd.read_parquet(PATHS.training_path)
    return frame[[column for column in columns if column in frame.columns]]


@st.cache_resource(show_spinner="Loading Gold model...")
def load_model_package() -> dict:
    return joblib.load(PATHS.model_path)

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import driver_position_chart, probability_chart
from src.dashboard.data import load_model_package
from src.dashboard.inference import probability_frame


def _label_for_session(row: pd.Series) -> str:
    return f"{int(row.year)} | {row.event_type} | {row.circuit_short_name} | session {int(row.session_key)}"


def render_predictor(model_df: pd.DataFrame) -> None:
    model_package = load_model_package()
    sessions = (
        model_df[["session_key", "year", "event_type", "circuit_short_name"]]
        .drop_duplicates("session_key")
        .sort_values(["year", "session_key"])
    )
    session_options = {_label_for_session(row): int(row.session_key) for _, row in sessions.iterrows()}
    selected_session_label = st.selectbox("Session", list(session_options.keys()), index=len(session_options) - 1)
    session_key = session_options[selected_session_label]

    session_df = model_df[model_df["session_key"].eq(session_key)].copy()
    drivers = (
        session_df[["driver_number", "full_name", "name_acronym", "team_name"]]
        .drop_duplicates("driver_number")
        .sort_values("driver_number")
    )
    driver_options = {
        f"{int(row.driver_number)} | {row.name_acronym} | {row.full_name} | {row.team_name}": int(row.driver_number)
        for _, row in drivers.iterrows()
    }
    selected_driver_label = st.selectbox("Driver", list(driver_options.keys()))
    driver_number = driver_options[selected_driver_label]

    driver_df = session_df[session_df["driver_number"].eq(driver_number)].sort_values("lap_number")
    min_lap = int(driver_df["lap_number"].min())
    max_lap = int(driver_df["lap_number"].max())
    selected_lap = st.slider("Lap", min_lap, max_lap, min(max_lap, max(min_lap, 10)))
    selected = driver_df[driver_df["lap_number"].eq(selected_lap)].tail(1)
    if selected.empty:
        st.warning("No Gold row exists for the selected lap.")
        return

    probabilities = probability_frame(model_package, selected)
    prediction = probabilities.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prediction", prediction["finish_bucket"])
    col2.metric("Probability", f"{prediction['probability']:.1%}")
    col3.metric("Current Position", int(selected["current_position"].iloc[0]))
    col4.metric("Race Progress", f"{selected['race_progress_pct'].iloc[0]:.1%}")

    st.plotly_chart(probability_chart(probabilities), use_container_width=True)

    state_columns = [
        "lap_number",
        "current_position",
        "compound",
        "current_tyre_age",
        "pit_count_so_far",
        "laps_since_last_pit",
        "track_temperature",
        "gap_to_leader_seconds",
        "target_finish_bucket_label",
    ]
    st.dataframe(selected[state_columns], use_container_width=True, hide_index=True)
    st.plotly_chart(driver_position_chart(driver_df, selected_lap), use_container_width=True)

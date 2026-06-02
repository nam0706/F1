from __future__ import annotations

import streamlit as st

from src.dashboard.data import load_model_frame, load_training_data
from src.dashboard.pages.overview import render_overview
from src.dashboard.pages.predictor import render_predictor
from src.dashboard.paths import PATHS


def main() -> None:
    st.set_page_config(
        page_title="F1 Gold Predictor Demo",
        page_icon="F1",
        layout="wide",
    )
    st.title("F1 Gold Predictor Demo")
    st.caption("Gold-contract model demo using data/gold artifacts only.")

    missing = [str(path.relative_to(PATHS.project_root)) for path in PATHS.required_artifacts if not path.exists()]
    if missing:
        st.error("Missing required artifacts: " + ", ".join(missing))
        st.stop()

    training_df = load_training_data()
    model_df = load_model_frame()
    overview_tab, predictor_tab = st.tabs(["Gold Overview", "Race Predictor"])
    with overview_tab:
        render_overview(training_df)
    with predictor_tab:
        render_predictor(model_df)

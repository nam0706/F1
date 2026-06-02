from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import target_distribution_chart
from src.dashboard.data import read_json
from src.dashboard.paths import PATHS


def _status_badge(status: str) -> str:
    return "PASS" if status == "PASS" else "REVIEW"


def render_overview(training_df: pd.DataFrame) -> None:
    leakage = read_json(PATHS.leakage_path)
    feature_contract = read_json(PATHS.feature_contract_path)
    training_contract = read_json(PATHS.training_contract_path)
    model_metadata = read_json(PATHS.model_metadata_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gold Rows", f"{len(training_df):,}")
    col2.metric("Feature Columns", f"{model_metadata['feature_count']:,}")
    col3.metric("Leakage", _status_badge(leakage.get("status", "UNKNOWN")))
    col4.metric("Macro F1", f"{model_metadata['metrics']['f1_macro']:.3f}")

    st.plotly_chart(target_distribution_chart(training_df), use_container_width=True)

    horizon_rows = pd.DataFrame(
        [
            {
                "dataset": name,
                "rows": meta["rows"],
                "unique_driver_sessions": meta["unique_driver_sessions"],
            }
            for name, meta in training_contract["datasets"].items()
        ]
    )
    st.dataframe(horizon_rows, use_container_width=True, hide_index=True)

    with st.expander("Feature contract whitelist", expanded=False):
        whitelist = []
        for group, contract in feature_contract.get("feature_tables", {}).items():
            for column in contract.get("model_allowed_columns", []):
                whitelist.append(
                    {
                        "group": group,
                        "column": column,
                        "availability_rule": contract.get("availability_rule"),
                    }
                )
        st.dataframe(pd.DataFrame(whitelist), use_container_width=True, hide_index=True)

from __future__ import annotations

import pandas as pd
import plotly.express as px


def target_distribution_chart(training_df: pd.DataFrame):
    target_counts = (
        training_df.groupby("target_finish_bucket_label")
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    fig = px.bar(
        target_counts,
        x="target_finish_bucket_label",
        y="rows",
        color="target_finish_bucket_label",
        title="Gold Training Target Distribution",
        labels={"target_finish_bucket_label": "Finish bucket", "rows": "Rows"},
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=90))
    return fig


def probability_chart(probabilities: pd.DataFrame):
    ordered = probabilities.sort_values("class_id")
    fig = px.bar(
        ordered,
        x="finish_bucket",
        y="probability",
        color="finish_bucket",
        text=ordered["probability"].map(lambda value: f"{value:.1%}"),
        title="Predicted Finish Bucket Probabilities",
        labels={"finish_bucket": "Finish bucket", "probability": "Probability"},
    )
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=50, b=90))
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def driver_position_chart(driver_df: pd.DataFrame, selected_lap: int):
    pace = driver_df[
        ["lap_number", "current_position", "race_progress_pct", "current_tyre_age", "track_temperature"]
    ].copy()
    fig = px.line(
        pace,
        x="lap_number",
        y="current_position",
        title="Driver Live Position by Lap",
        labels={"lap_number": "Lap", "current_position": "Position"},
    )
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=selected_lap, line_dash="dash", line_color="red")
    return fig

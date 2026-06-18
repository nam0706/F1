from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from replay_data import available_location_sessions, build_replay_payload


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
MASTER_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "feature_engineering_metadata.json"
AUDIT_PATH = PROJECT_ROOT / "reports" / "data_quality" / "master_dataset_audit.csv"
CLEANING_SUMMARY_PATH = PROJECT_ROOT / "reports" / "data_quality" / "cleaning_summary.csv"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
LOCATION_PATH = PROJECT_ROOT / "data" / "cleaned" / "location.parquet"
SESSIONS_PATH = PROJECT_ROOT / "data" / "cleaned" / "sessions.csv"
REPLAY_TEMPLATE_PATH = APP_DIR / "replay_assets" / "race_replay.html"
REPLAY_PAYLOAD_VERSION = "2026-06-17-sector-track-colors-v2"
FINAL_SESSION_TYPES = ["Race", "Sprint", "Qualifying"]
CSV_ENDPOINTS = [
    "sessions",
    "meetings",
    "drivers",
    "session_results",
    "starting_grid",
    "laps",
    "stints",
    "weather",
    "intervals",
    "position",
    "race_control",
    "pit",
    "overtakes",
    "team_radio",
]
PARQUET_ENDPOINTS = ["car_data", "location"]

KEY_COLUMNS = ["session_key", "driver_number", "lap_number"]
TARGET_COLUMNS = ["target_win", "target_podium", "target_top10"]
BLOCKED_CURRENT_LAP_COLUMNS = [
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
]


st.set_page_config(
    page_title="F1 Final Dataset Explorer",
    page_icon="F1",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    if MASTER_PARQUET_PATH.exists():
        df = pd.read_parquet(MASTER_PARQUET_PATH)
    elif MASTER_PATH.exists():
        df = pd.read_csv(MASTER_PATH, low_memory=False)
    else:
        return pd.DataFrame()
    return add_display_session_type(df)


def add_display_session_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "session_key" not in df.columns or not SESSIONS_PATH.exists():
        return df
    sessions = pd.read_csv(SESSIONS_PATH, usecols=["session_key", "session_name", "session_type"])
    sessions = sessions.rename(
        columns={
            "session_type": "raw_session_type",
            "session_name": "raw_session_name",
        }
    )
    result = df.merge(sessions, on="session_key", how="left")
    result["session_type"] = result["session_type"].where(
        result["raw_session_name"].ne("Sprint"),
        "Sprint",
    )
    return result.drop(columns=["raw_session_type", "raw_session_name"], errors="ignore")


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_clean_csv(name: str, usecols: tuple[str, ...] | None = None, nrows: int | None = None) -> pd.DataFrame:
    path = CLEAN_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    kwargs = {"low_memory": False}
    if usecols is not None:
        kwargs["usecols"] = lambda col: col in set(usecols)
    if nrows is not None:
        kwargs["nrows"] = nrows
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_endpoint_summary() -> pd.DataFrame:
    grain_map = {
        "sessions": "session",
        "meetings": "meeting",
        "drivers": "driver-session",
        "session_results": "driver-session result",
        "starting_grid": "driver-session pre-session",
        "laps": "driver-lap",
        "stints": "driver-stint",
        "weather": "session time-series",
        "intervals": "driver time-series",
        "position": "driver time-series",
        "race_control": "session event",
        "pit": "driver event",
        "overtakes": "event",
        "team_radio": "driver event",
        "car_data": "driver telemetry",
        "location": "driver location",
    }
    rows = []
    for name in CSV_ENDPOINTS:
        path = CLEAN_DIR / f"{name}.csv"
        if not path.exists():
            rows.append({"endpoint": name, "grain": grain_map.get(name, "unknown"), "rows": 0, "cols": 0, "null_pct": None})
            continue
        with path.open("rb") as handle:
            row_count = max(sum(1 for _ in handle) - 1, 0)
        sample = load_clean_csv(name, nrows=200_000)
        rows.append(
            {
                "endpoint": name,
                "grain": grain_map.get(name, "unknown"),
                "rows": row_count,
                "cols": len(sample.columns),
                "null_pct": round(sample.isna().mean().mean() * 100, 2) if not sample.empty else None,
                "sessions": sample["session_key"].nunique() if "session_key" in sample.columns else None,
                "drivers": sample["driver_number"].nunique() if "driver_number" in sample.columns else None,
            }
        )
    for name in PARQUET_ENDPOINTS:
        path = CLEAN_DIR / f"{name}.parquet"
        if not path.exists():
            rows.append({"endpoint": name, "grain": grain_map.get(name, "unknown"), "rows": 0, "cols": 0, "null_pct": None})
            continue
        parquet_file = pq.ParquetFile(path)
        rows.append(
            {
                "endpoint": name,
                "grain": grain_map.get(name, "unknown"),
                "rows": parquet_file.metadata.num_rows,
                "cols": len(parquet_file.schema_arrow.names),
                "null_pct": None,
                "sessions": None,
                "drivers": None,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_location_sample(session_key: int, max_rows: int = 120_000) -> pd.DataFrame:
    path = CLEAN_DIR / "location.parquet"
    if not path.exists():
        return pd.DataFrame()
    dataset = ds.dataset(path, format="parquet")
    cols = [col for col in ["session_key", "driver_number", "lap_number", "X", "Y", "Z"] if col in dataset.schema.names]
    table = dataset.to_table(columns=cols, filter=ds.field("session_key") == int(session_key))
    loc = table.to_pandas()
    if len(loc) > max_rows:
        loc = loc.sample(max_rows, random_state=42)
    return loc


@st.cache_data(show_spinner=False)
def load_car_data_sample(session_key: int, max_rows: int = 120_000) -> pd.DataFrame:
    path = CLEAN_DIR / "car_data.parquet"
    if not path.exists():
        return pd.DataFrame()
    dataset = ds.dataset(path, format="parquet")
    cols = [
        col
        for col in ["session_key", "driver_number", "lap_number", "Speed", "RPM", "Throttle", "Brake", "DRS", "nGear"]
        if col in dataset.schema.names
    ]
    table = dataset.to_table(columns=cols, filter=ds.field("session_key") == int(session_key))
    tele = table.to_pandas()
    if len(tele) > max_rows:
        tele = tele.sample(max_rows, random_state=42)
    return tele


@st.cache_data(show_spinner=False)
def load_replay_payload(
    session_key: int,
    frame_step_seconds: int,
    selected_drivers: tuple[int, ...],
    driver_labels: dict[int, str],
    max_frames: int,
    payload_version: str,
) -> dict:
    _ = payload_version
    return build_replay_payload(
        session_key=session_key,
        selected_drivers=selected_drivers,
        driver_labels=driver_labels,
        frame_step_seconds=frame_step_seconds,
        max_frames=max_frames,
    )


@st.cache_data(show_spinner=False)
def load_available_location_sessions() -> set[int]:
    return available_location_sessions()


def existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "Identifiers": existing(df, KEY_COLUMNS + ["meeting_key", "year", "session_type", "circuit_key"]),
        "Session and driver": existing(
            df,
            [
                "circuit_short_name",
                "country_name",
                "full_name",
                "name_acronym",
                "team_name",
                "date_start",
                "date_end",
            ],
        ),
        "Lap pace": existing(
            df,
            [
                "lap_duration",
                "duration_sector_1",
                "duration_sector_2",
                "duration_sector_3",
                "i1_speed",
                "i2_speed",
                "st_speed",
            ],
        ),
        "Rolling pace": [c for c in df.columns if c.startswith("rolling_")]
        + existing(df, ["lap_delta_prev", "laps_to_go", "race_completion_pct"]),
        "Tyre and stint": existing(
            df,
            ["stint_number", "compound", "current_tyre_age", "compound_cliff", "laps_until_cliff", "is_past_cliff"],
        ),
        "Weather": existing(
            df,
            ["air_temperature", "track_temperature", "humidity", "rainfall", "wind_speed", "track_temp_evolution"],
        ),
        "Pit before lap": [c for c in df.columns if c.startswith("pit_") and "before_lap" in c]
        + existing(df, ["was_pit_previous_lap", "laps_since_last_pit"]),
        "Race-control before lap": [
            c
            for c in df.columns
            if c.startswith(("race_control_", "yellow_flag_", "green_flag_")) and "before_lap" in c
        ]
        + existing(df, ["yellow_flag_previous_lap"]),
        "Overtakes": [c for c in df.columns if c.startswith(("overtakes_", "net_overtakes_"))],
        "Telemetry previous lap": [c for c in df.columns if c.startswith("tele_prev_")],
        "Location previous lap": [c for c in df.columns if c.startswith("loc_prev_")],
        "Targets and outcomes": existing(df, ["final_position", "status"] + TARGET_COLUMNS),
        "Current-lap diagnostics": existing(df, BLOCKED_CURRENT_LAP_COLUMNS),
    }


def group_coverage(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for group, cols in groups.items():
        if not cols:
            rows.append({"group": group, "columns": 0, "row_coverage": None, "avg_missing": None})
            continue
        rows.append(
            {
                "group": group,
                "columns": len(cols),
                "row_coverage": df[cols].notna().any(axis=1).mean(),
                "avg_missing": df[cols].isna().mean().mean(),
            }
        )
    return pd.DataFrame(rows)


def make_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    filtered = df
    if "session_type" in filtered.columns:
        available_final_types = [session_type for session_type in FINAL_SESSION_TYPES if session_type in set(filtered["session_type"].dropna())]
        if available_final_types:
            filtered = filtered[filtered["session_type"].isin(available_final_types)].copy()

    if "year" in df.columns:
        years = sorted(filtered["year"].dropna().unique().tolist())
        selected_years = st.sidebar.multiselect("Year", years, default=years)
        if selected_years:
            filtered = filtered[filtered["year"].isin(selected_years)]

    if "session_type" in filtered.columns:
        available_types = set(filtered["session_type"].dropna())
        default_types = [session_type for session_type in FINAL_SESSION_TYPES if session_type in available_types]
        selected_types = st.sidebar.multiselect("Session type", FINAL_SESSION_TYPES, default=default_types)
        filtered = filtered[filtered["session_type"].isin(selected_types)]

    if "circuit_short_name" in filtered.columns:
        circuits = sorted(filtered["circuit_short_name"].dropna().unique().tolist())
        selected_circuit = st.sidebar.selectbox("Circuit", ["All"] + circuits)
        if selected_circuit != "All":
            filtered = filtered[filtered["circuit_short_name"] == selected_circuit]

    return filtered


def metric_row(df: pd.DataFrame, metadata: dict) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Sessions", f"{df['session_key'].nunique():,}" if "session_key" in df.columns else "n/a")
    col4.metric("Drivers", f"{df['driver_number'].nunique():,}" if "driver_number" in df.columns else "n/a")
    col5.metric("Model-safe features", f"{len(metadata.get('model_feature_columns', [])):,}")


def overview_tab(df: pd.DataFrame, filtered: pd.DataFrame, metadata: dict) -> None:
    st.subheader("Final Dataset Overview")
    metric_row(df, metadata)

    st.markdown(
        f"""
        **Prediction contract:** `{metadata.get("prediction_contract", "unknown")}`  
        **Grain:** `session_key + driver_number + lap_number`  
        **Filtered rows:** `{len(filtered):,}`
        """
    )

    targets = existing(filtered, TARGET_COLUMNS)
    if targets:
        target_rates = filtered[targets].mean(numeric_only=True).reset_index()
        target_rates.columns = ["target", "rate"]
        fig = px.bar(target_rates, x="target", y="rate", text=target_rates["rate"].map(pct), title="Target Rates")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    if {"year", "session_type"}.issubset(filtered.columns):
        sessions = filtered.drop_duplicates("session_key").groupby(["year", "session_type"]).size().reset_index(name="sessions")
        fig = px.bar(sessions, x="year", y="sessions", color="session_type", barmode="group", title="Sessions by Year and Type")
        st.plotly_chart(fig, use_container_width=True)


def quality_tab(df: pd.DataFrame, metadata: dict, audit: pd.DataFrame, cleaning_summary: pd.DataFrame) -> None:
    st.subheader("Data Quality")
    duplicate_rows = int(df.duplicated(KEY_COLUMNS).sum()) if set(KEY_COLUMNS).issubset(df.columns) else None
    blocked = set(existing(df, BLOCKED_CURRENT_LAP_COLUMNS + ["final_position"] + TARGET_COLUMNS))
    model_features = set(metadata.get("model_feature_columns", []))
    leakage = sorted(blocked & model_features)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duplicate grain rows", "n/a" if duplicate_rows is None else duplicate_rows)
    col2.metric("Leakage columns in model feature list", len(leakage))
    col3.metric("Audit checks", len(audit) if not audit.empty else "n/a")
    col4.metric("Validation warnings", int(audit.loc[audit["check"] == "cleaned_validation_warnings", "value"].iloc[0]) if not audit.empty and (audit["check"] == "cleaned_validation_warnings").any() else "n/a")

    if leakage:
        st.error(f"Blocked columns in model features: {leakage}")
    else:
        st.success("No target/outcome/current-lap diagnostic columns in metadata model feature list.")

    if not audit.empty:
        st.dataframe(audit, use_container_width=True)

    groups = feature_groups(df)
    coverage = group_coverage(df, groups)
    fig = px.bar(
        coverage.dropna(subset=["row_coverage"]),
        y="group",
        x="row_coverage",
        orientation="h",
        text=coverage.dropna(subset=["row_coverage"])["row_coverage"].map(pct),
        title="Feature Group Coverage",
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(coverage, use_container_width=True)

    if not cleaning_summary.empty:
        st.subheader("Cleaning Summary")
        st.dataframe(cleaning_summary, use_container_width=True)


def feature_groups_tab(df: pd.DataFrame, metadata: dict) -> None:
    st.subheader("Feature Groups")
    groups = feature_groups(df)
    selected_group = st.selectbox("Feature group", list(groups.keys()))
    cols = groups[selected_group]
    st.write(f"{selected_group}: {len(cols)} columns")
    st.dataframe(pd.DataFrame({"column": cols}), use_container_width=True)

    model_features = set(metadata.get("model_feature_columns", []))
    group_feature_status = pd.DataFrame(
        {
            "column": cols,
            "in_model_safe_list": [col in model_features for col in cols],
            "missing_rate": [df[col].isna().mean() if col in df.columns else None for col in cols],
            "dtype": [str(df[col].dtype) if col in df.columns else "" for col in cols],
        }
    )
    st.dataframe(group_feature_status, use_container_width=True)


def live_contract_tab(df: pd.DataFrame, metadata: dict) -> None:
    st.subheader("Live before lap N Contract")
    st.markdown(
        """
        For a row at lap `N`, safe features may only use information available before lap `N` starts.

        Safe examples:
        - `*_before_lap`
        - `tele_prev_*`
        - `loc_prev_*`
        - shifted rolling pace features

        Blocked for model input:
        - `target_*`
        - `final_position`
        - `*_current_lap`
        """
    )

    if not set(KEY_COLUMNS).issubset(df.columns):
        st.warning("Key columns are missing.")
        return
    if df.empty:
        st.warning("No rows after sidebar filters.")
        return

    session_keys = sorted(df["session_key"].dropna().unique().tolist())
    selected_session = st.selectbox("Session key", session_keys[:500])
    session_df = df[df["session_key"] == selected_session]
    drivers = sorted(session_df["driver_number"].dropna().unique().tolist())
    selected_driver = st.selectbox("Driver number", drivers)
    driver_df = session_df[session_df["driver_number"] == selected_driver]
    laps = sorted(driver_df["lap_number"].dropna().unique().tolist())
    selected_lap = st.selectbox("Lap N", laps)
    row = driver_df[driver_df["lap_number"] == selected_lap].head(1)

    allowed_cols = [
        col
        for col in metadata.get("model_feature_columns", [])
        if col in row.columns and ("before_lap" in col or col.startswith(("tele_prev_", "loc_prev_", "rolling_")) or col in ["lap_delta_prev", "laps_to_go", "race_completion_pct"])
    ]
    blocked_cols = existing(row, ["final_position"] + TARGET_COLUMNS + BLOCKED_CURRENT_LAP_COLUMNS)

    st.write("Selected row")
    st.dataframe(row[existing(row, KEY_COLUMNS + ["year", "session_type", "circuit_short_name", "full_name", "team_name", "compound"])], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Safe feature examples")
        st.dataframe(row[allowed_cols[:30]].T.rename(columns={row.index[0]: "value"}) if allowed_cols else pd.DataFrame(), use_container_width=True)
    with col2:
        st.write("Blocked diagnostic/outcome columns")
        st.dataframe(row[blocked_cols].T.rename(columns={row.index[0]: "value"}) if blocked_cols else pd.DataFrame(), use_container_width=True)


def eda_tab(filtered: pd.DataFrame) -> None:
    st.subheader("EDA Charts")
    if filtered.empty:
        st.warning("No rows after filters.")
        return

    chart = st.selectbox(
        "Chart",
        [
            "Endpoint overview",
            "Duplicate key audit",
            "Result distribution",
            "Grid conversion",
            "Lap duration distribution",
            "Lap duration outliers",
            "Lap telemetry correlation",
            "Target distribution",
            "Target win by race progress",
            "Feature missingness",
            "Final dataset correlation matrix",
            "Top correlations with target_win",
            "Rolling pace validation",
            "Tyre age vs lap duration",
            "Tyre degradation by compound",
            "Compound rows and cliff",
            "Weather vs lap duration",
            "Weather correlation",
            "Weather profile",
            "Pit stops before lap vs target",
            "Race-control context",
            "Race-control breakdown",
            "Pit strategy vs final position",
            "Pit duration distribution",
            "Telemetry distributions",
            "Telemetry coverage and speed",
            "Telemetry vs lap duration",
            "Location spread",
            "Location feature relationship",
            "Raw location sample",
            "Telemetry scatter",
            "Overtakes",
            "Overtakes vs target",
            "Team radio count",
        ],
    )

    sample = filtered.sample(min(len(filtered), 25_000), random_state=42) if len(filtered) > 25_000 else filtered

    if chart == "Endpoint overview":
        summary = load_endpoint_summary()
        if summary.empty:
            st.info("No endpoint summary available.")
        else:
            left, right = st.columns(2)
            fig = px.bar(summary, y="endpoint", x="rows", orientation="h", title="Endpoint Row Counts")
            fig.update_xaxes(type="log")
            left.plotly_chart(fig, use_container_width=True)
            fig = px.bar(summary.dropna(subset=["null_pct"]), y="endpoint", x="null_pct", orientation="h", title="Endpoint Missingness")
            right.plotly_chart(fig, use_container_width=True)
            st.dataframe(summary, use_container_width=True)

    elif chart == "Duplicate key audit":
        key_specs = {
            "sessions": ["session_key"],
            "meetings": ["meeting_key", "year"],
            "drivers": ["session_key", "driver_number"],
            "session_results": ["session_key", "driver_number"],
            "starting_grid": ["session_key", "driver_number"],
            "laps": ["session_key", "driver_number", "lap_number"],
            "stints": ["session_key", "driver_number", "stint_number"],
            "weather": ["session_key", "date"],
            "intervals": ["session_key", "driver_number", "date"],
            "position": ["session_key", "driver_number", "date"],
            "pit": ["session_key", "driver_number", "lap_number", "date"],
            "team_radio": ["session_key", "driver_number", "date"],
        }
        audit_rows = []
        for name, keys in key_specs.items():
            df_ep = load_clean_csv(name)
            keys = [k for k in keys if k in df_ep.columns]
            if df_ep.empty or not keys:
                continue
            audit_rows.append(
                {
                    "endpoint": name,
                    "keys": " + ".join(keys),
                    "rows": len(df_ep),
                    "unique_keys": df_ep[keys].drop_duplicates().shape[0],
                    "duplicate_rows_on_key": int(df_ep.duplicated(keys).sum()),
                }
            )
        audit_df = pd.DataFrame(audit_rows)
        if audit_df.empty:
            st.info("No duplicate audit data available.")
        else:
            fig = px.bar(audit_df, x="endpoint", y="duplicate_rows_on_key", title="Duplicate Rows On Declared Join Keys")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(audit_df, use_container_width=True)

    elif chart == "Result distribution" and not load_clean_csv("session_results").empty:
        results = load_clean_csv("session_results")
        if "position" in results.columns:
            pos = pd.to_numeric(results["position"], errors="coerce")
            counts = pos.dropna().astype(int).value_counts().sort_index().reset_index()
            counts.columns = ["position", "count"]
            fig = px.bar(counts, x="position", y="count", title="Final Position Distribution")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Grid conversion":
        results = load_clean_csv("session_results")
        grid = load_clean_csv("starting_grid")
        if not results.empty and "position" in results.columns:
            res = results.copy()
            res["position"] = pd.to_numeric(res["position"], errors="coerce")
            if not grid.empty and {"session_key", "driver_number", "position"}.issubset(grid.columns):
                grid2 = grid[["session_key", "driver_number", "position"]].rename(columns={"position": "grid_position"})
                res = res.merge(grid2, on=["session_key", "driver_number"], how="left")
                res["grid_position"] = pd.to_numeric(res["grid_position"], errors="coerce")
                res["position_gain"] = res["grid_position"] - res["position"]
            if "grid_position" in res.columns:
                fig = px.scatter(res, x="grid_position", y="position", title="Grid Position vs Final Position", opacity=0.45)
                st.plotly_chart(fig, use_container_width=True)

    if chart == "Lap duration distribution" and "lap_duration" in sample.columns:
        fig = px.histogram(sample, x="lap_duration", nbins=80, color="session_type" if "session_type" in sample.columns else None)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Lap duration outliers" and "lap_duration" in filtered.columns:
        col1, col2 = st.columns(2)
        fig = px.box(filtered.dropna(subset=["lap_duration"]), x="lap_duration", points=False, title="Lap Duration Boxplot")
        col1.plotly_chart(fig, use_container_width=True)
        if "is_outlier_lap" in filtered.columns:
            fig = px.histogram(
                filtered.dropna(subset=["lap_duration"]),
                x="lap_duration",
                color="is_outlier_lap",
                nbins=80,
                title="Normal vs Outlier Laps",
            )
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col2.info("Column `is_outlier_lap` is missing.")

    elif chart == "Lap telemetry correlation":
        corr_cols = existing(filtered, ["lap_duration", "i1_speed", "i2_speed", "st_speed", "duration_sector_1", "duration_sector_2", "duration_sector_3"])
        if len(corr_cols) >= 3:
            corr = filtered[corr_cols].dropna().sample(min(len(filtered[corr_cols].dropna()), 5000), random_state=42).corr(method="spearman")
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Spearman Correlation - Lap Telemetry")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough lap telemetry columns.")

    elif chart == "Target distribution":
        targets = existing(filtered, TARGET_COLUMNS)
        if targets:
            target_rates = filtered[targets].mean(numeric_only=True).reset_index()
            target_rates.columns = ["target", "positive_rate"]
            col1, col2 = st.columns(2)
            fig = px.bar(target_rates, x="target", y="positive_rate", text=target_rates["positive_rate"].map(pct), title="Target Positive Rates")
            fig.update_yaxes(tickformat=".0%")
            col1.plotly_chart(fig, use_container_width=True)
            long = filtered[targets].melt(var_name="target", value_name="label").dropna()
            fig = px.histogram(long, x="label", color="target", barmode="group", title="Target Label Counts")
            col2.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target columns are missing.")

    elif chart == "Target win by race progress" and {"race_completion_pct", "target_win"}.issubset(filtered.columns):
        tmp = filtered.dropna(subset=["race_completion_pct", "target_win"]).copy()
        tmp["progress_bin"] = pd.cut(tmp["race_completion_pct"], bins=10, include_lowest=True)
        progress = tmp.groupby("progress_bin", observed=True)["target_win"].mean().reset_index()
        progress["progress_mid"] = progress["progress_bin"].apply(lambda value: value.mid)
        fig = px.line(progress, x="progress_mid", y="target_win", markers=True, title="Target Win Rate by Race Progress")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Feature missingness":
        miss = (filtered.isna().mean() * 100).sort_values(ascending=False).head(30).reset_index()
        miss.columns = ["column", "missing_pct"]
        fig = px.bar(miss, y="column", x="missing_pct", orientation="h", title="Top Missing Features")
        fig.add_vline(x=20, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Final dataset correlation matrix":
        numeric = filtered.select_dtypes(include="number").copy()
        numeric = numeric[[col for col in numeric.columns if not col.startswith(("is_invalid_", "is_missing_"))]]
        if numeric.shape[1] < 2:
            st.info("Not enough numeric columns for correlation analysis.")
        else:
            corr = numeric.corr(method="spearman")
            pair_mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
            for i, col_a in enumerate(corr.columns):
                for col_b in corr.columns[i + 1:]:
                    pair_mask.loc[col_a, col_b] = True
            pairs = corr.where(pair_mask).stack().reset_index()
            pairs.columns = ["feature_a", "feature_b", "spearman_rho"]
            pairs["abs_rho"] = pairs["spearman_rho"].abs()
            pairs = pairs.sort_values("abs_rho", ascending=False)

            if "target_win" in corr.columns:
                heatmap_cols = corr["target_win"].abs().sort_values(ascending=False).head(25).index.tolist()
            else:
                heatmap_cols = numeric.var(numeric_only=True).sort_values(ascending=False).head(25).index.tolist()

            fig = px.imshow(
                corr.loc[heatmap_cols, heatmap_cols],
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Final Dataset Spearman Correlation Matrix - Top Numeric Features",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.write("Strongest numeric feature pairs")
            st.dataframe(pairs.head(100), use_container_width=True)
            with st.expander("Full correlation matrix"):
                st.dataframe(corr.round(3), use_container_width=True)

    elif chart == "Top correlations with target_win" and "target_win" in filtered.columns:
        numeric = filtered.select_dtypes(include="number")
        corr = numeric.corr(method="spearman")["target_win"].drop("target_win", errors="ignore").dropna().abs().sort_values(ascending=False).head(25)
        fig = px.bar(corr.reset_index(), y="index", x="target_win", orientation="h", title="Top Absolute Spearman Correlations With target_win")
        fig.update_layout(yaxis_title="", xaxis_title="|rho|")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Rolling pace validation" and {"lap_number", "lap_duration", "rolling_avg_lap_3", "driver_number", "session_key"}.issubset(filtered.columns):
        plot_df = filtered.dropna(subset=["lap_duration", "rolling_avg_lap_3"]).copy()
        if not plot_df.empty:
            driver = plot_df["driver_number"].value_counts().index[0]
            session = plot_df.loc[plot_df["driver_number"] == driver, "session_key"].value_counts().index[0]
            driver_df = plot_df[(plot_df["driver_number"] == driver) & (plot_df["session_key"] == session)].sort_values("lap_number")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=driver_df["lap_number"], y=driver_df["lap_duration"], mode="lines+markers", name="Actual lap"))
            fig.add_trace(go.Scatter(x=driver_df["lap_number"], y=driver_df["rolling_avg_lap_3"], mode="lines", name="Rolling avg 3"))
            if "rolling_avg_lap_5" in driver_df.columns:
                fig.add_trace(go.Scatter(x=driver_df["lap_number"], y=driver_df["rolling_avg_lap_5"], mode="lines", name="Rolling avg 5"))
            fig.update_layout(title=f"Rolling Feature Validation - Driver {driver}, Session {session}", xaxis_title="Lap", yaxis_title="Lap duration")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Tyre age vs lap duration" and {"current_tyre_age", "lap_duration"}.issubset(sample.columns):
        fig = px.scatter(
            sample.dropna(subset=["current_tyre_age", "lap_duration"]),
            x="current_tyre_age",
            y="lap_duration",
            color="compound" if "compound" in sample.columns else None,
            opacity=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Tyre degradation by compound" and {"current_tyre_age", "lap_duration", "compound"}.issubset(filtered.columns):
        tmp = filtered.dropna(subset=["current_tyre_age", "lap_duration", "compound"]).copy()
        tmp = tmp[tmp["current_tyre_age"].between(0, 55)]
        perf = tmp.groupby(["compound", "current_tyre_age"], observed=True)["lap_duration"].median().reset_index()
        fig = px.line(perf, x="current_tyre_age", y="lap_duration", color="compound", markers=False, title="Median Lap Duration by Tyre Age")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Compound rows and cliff":
        col1, col2, col3 = st.columns(3)
        if "compound" in filtered.columns:
            counts = filtered["compound"].value_counts().reset_index()
            counts.columns = ["compound", "rows"]
            fig = px.bar(counts, x="compound", y="rows", title="Rows by Compound")
            col1.plotly_chart(fig, use_container_width=True)
        if {"current_tyre_age", "lap_duration", "compound"}.issubset(filtered.columns):
            tmp = filtered.dropna(subset=["current_tyre_age", "lap_duration", "compound"])
            perf = tmp.groupby(["compound", "current_tyre_age"], observed=True)["lap_duration"].median().reset_index()
            fig = px.line(perf, x="current_tyre_age", y="lap_duration", color="compound", title="Compound Degradation")
            col2.plotly_chart(fig, use_container_width=True)
        if "laps_until_cliff" in filtered.columns:
            fig = px.histogram(filtered.dropna(subset=["laps_until_cliff"]), x="laps_until_cliff", nbins=50, title="Laps Until Cliff")
            col3.plotly_chart(fig, use_container_width=True)

    elif chart == "Weather vs lap duration" and {"track_temperature", "lap_duration"}.issubset(sample.columns):
        fig = px.scatter(
            sample.dropna(subset=["track_temperature", "lap_duration"]),
            x="track_temperature",
            y="lap_duration",
            color="rainfall" if "rainfall" in sample.columns else None,
            opacity=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Weather correlation":
        weather_cols = existing(filtered, ["air_temperature", "track_temperature", "humidity", "rainfall", "wind_speed", "track_temp_evolution", "lap_duration", "target_win"])
        if len(weather_cols) >= 3:
            fig = px.imshow(filtered[weather_cols].corr(numeric_only=True, method="spearman"), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Weather Spearman Correlation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough weather columns.")

    elif chart == "Weather profile":
        weather = load_clean_csv("weather")
        if weather.empty:
            st.info("weather.csv is missing or empty.")
        else:
            cols = existing(weather, ["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall"])
            if cols:
                long = weather[cols].sample(min(len(weather), 100_000), random_state=42).melt(var_name="feature", value_name="value").dropna()
                fig = px.histogram(long, x="value", color="feature", facet_col="feature", facet_col_wrap=3, nbins=40, title="Weather Distributions")
                fig.update_yaxes(matches=None)
                fig.update_xaxes(matches=None)
                st.plotly_chart(fig, use_container_width=True)
            if {"session_key", "date", "track_temperature"}.issubset(weather.columns):
                weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
                session_key = int(weather["session_key"].value_counts().index[0])
                w = weather[weather["session_key"] == session_key].sort_values("date")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=w["date"], y=w["track_temperature"], mode="lines", name="track"))
                if "air_temperature" in w.columns:
                    fig.add_trace(go.Scatter(x=w["date"], y=w["air_temperature"], mode="lines", name="air"))
                fig.update_layout(title=f"Temperature Evolution - Session {session_key}", xaxis_title="Time", yaxis_title="Temperature")
                st.plotly_chart(fig, use_container_width=True)

    elif chart == "Pit stops before lap vs target" and {"pit_stop_count_before_lap", "target_top10"}.issubset(filtered.columns):
        grouped = filtered.groupby("pit_stop_count_before_lap", dropna=False)[["target_win", "target_podium", "target_top10"]].mean().reset_index()
        fig = px.line(grouped, x="pit_stop_count_before_lap", y=existing(grouped, TARGET_COLUMNS), markers=True)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Race-control context":
        cols = existing(
            filtered,
            [
                "race_control_events_before_lap",
                "yellow_flag_events_before_lap",
                "yellow_flag_previous_lap",
                "lap_duration",
            ],
        )
        if {"yellow_flag_previous_lap", "lap_duration"}.issubset(filtered.columns):
            fig = px.box(filtered, x="yellow_flag_previous_lap", y="lap_duration", points=False)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered[cols].describe().T if cols else pd.DataFrame(), use_container_width=True)

    elif chart == "Race-control breakdown":
        race_control = load_clean_csv("race_control")
        if race_control.empty:
            st.info("race_control.csv is missing or empty.")
        else:
            col1, col2, col3 = st.columns(3)
            if "category" in race_control.columns:
                cat = race_control["category"].value_counts().head(12).reset_index()
                cat.columns = ["category", "count"]
                fig = px.bar(cat, y="category", x="count", orientation="h", title="Race Control Category Frequency")
                col1.plotly_chart(fig, use_container_width=True)
            if "flag" in race_control.columns:
                flag = race_control["flag"].replace("", pd.NA).dropna().value_counts().head(10).reset_index()
                flag.columns = ["flag", "count"]
                fig = px.bar(flag, y="flag", x="count", orientation="h", title="Flag Frequency")
                col2.plotly_chart(fig, use_container_width=True)
            lap_col = "lap_number_clean" if "lap_number_clean" in race_control.columns else "lap_number"
            if lap_col in race_control.columns:
                fig = px.histogram(race_control, x=lap_col, nbins=60, title="Race Control Events By Lap")
                col3.plotly_chart(fig, use_container_width=True)

    elif chart == "Pit strategy vs final position":
        cols = existing(filtered, ["pit_stop_count_before_lap", "laps_since_last_pit", "final_position", "target_win"])
        if "pit_stop_count_before_lap" in filtered.columns and "final_position" in filtered.columns:
            col1, col2 = st.columns(2)
            fig = px.box(filtered, x="pit_stop_count_before_lap", y="final_position", points=False, title="Pit Count Before Lap vs Final Position")
            fig.update_yaxes(autorange="reversed")
            col1.plotly_chart(fig, use_container_width=True)
            if {"laps_since_last_pit", "target_win"}.issubset(filtered.columns):
                tmp = filtered.dropna(subset=["laps_since_last_pit", "target_win"]).copy()
                tmp["pit_age_bin"] = pd.cut(tmp["laps_since_last_pit"], bins=10, include_lowest=True)
                rates = tmp.groupby("pit_age_bin", observed=True)["target_win"].mean().reset_index()
                rates["pit_age_mid"] = rates["pit_age_bin"].apply(lambda value: value.mid)
                fig = px.line(rates, x="pit_age_mid", y="target_win", markers=True, title="Win Target Rate by Laps Since Last Pit")
                fig.update_yaxes(tickformat=".0%")
                col2.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered[cols].describe().T if cols else pd.DataFrame(), use_container_width=True)
        else:
            st.info("Required pit/final position columns are missing.")

    elif chart == "Pit duration distribution":
        pit = load_clean_csv("pit")
        if pit.empty:
            st.info("pit.csv is missing or empty.")
        else:
            col1, col2, col3 = st.columns(3)
            if {"session_key", "driver_number"}.issubset(pit.columns):
                pit_count = pit.groupby(["session_key", "driver_number"]).size().reset_index(name="pit_count")
                fig = px.histogram(pit_count, x="pit_count", nbins=20, title="Pit Stops Per Driver-Session")
                col1.plotly_chart(fig, use_container_width=True)
            if "pit_duration" in pit.columns:
                tmp = pit.copy()
                tmp["pit_duration"] = pd.to_numeric(tmp["pit_duration"], errors="coerce").clip(upper=80)
                fig = px.histogram(tmp.dropna(subset=["pit_duration"]), x="pit_duration", nbins=50, title="Pit Duration Distribution")
                col2.plotly_chart(fig, use_container_width=True)
            if "lap_number" in pit.columns:
                fig = px.histogram(pit.dropna(subset=["lap_number"]), x="lap_number", nbins=60, title="Pit Stops By Lap")
                col3.plotly_chart(fig, use_container_width=True)

    elif chart == "Telemetry distributions":
        tele_cols = existing(filtered, ["tele_prev_speed_max", "tele_prev_speed_mean", "tele_prev_throttle_mean", "tele_prev_brake_mean", "tele_prev_rpm_mean", "tele_prev_ngear_mean"])
        if tele_cols:
            long = filtered[tele_cols].sample(min(len(filtered), 25000), random_state=42).melt(var_name="feature", value_name="value").dropna()
            fig = px.histogram(long, x="value", color="feature", facet_col="feature", facet_col_wrap=3, nbins=50, title="Telemetry Feature Distributions")
            fig.update_yaxes(matches=None)
            fig.update_xaxes(matches=None)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Telemetry columns are missing.")

    elif chart == "Telemetry coverage and speed" and "tele_prev_speed_max" in filtered.columns:
        col1, col2 = st.columns(2)
        col1.metric("tele_prev_speed_max coverage", pct(filtered["tele_prev_speed_max"].notna().mean()))
        fig = px.histogram(filtered.dropna(subset=["tele_prev_speed_max"]), x="tele_prev_speed_max", nbins=60)
        col2.plotly_chart(fig, use_container_width=True)

    elif chart == "Telemetry vs lap duration":
        col1, col2 = st.columns(2)
        if {"tele_prev_speed_mean", "lap_duration"}.issubset(sample.columns):
            fig = px.scatter(sample.dropna(subset=["tele_prev_speed_mean", "lap_duration"]), x="tele_prev_speed_mean", y="lap_duration", opacity=0.25, title="Previous Speed Mean vs Lap Duration")
            col1.plotly_chart(fig, use_container_width=True)
        if {"tele_prev_throttle_mean", "tele_prev_brake_mean"}.issubset(sample.columns):
            fig = px.scatter(sample.dropna(subset=["tele_prev_throttle_mean", "tele_prev_brake_mean"]), x="tele_prev_throttle_mean", y="tele_prev_brake_mean", color="target_win" if "target_win" in sample.columns else None, opacity=0.25, title="Throttle vs Brake Previous Lap")
            col2.plotly_chart(fig, use_container_width=True)

    elif chart == "Telemetry scatter":
        if "session_key" not in filtered.columns:
            st.info("session_key is missing.")
        else:
            session_key = int(filtered["session_key"].dropna().iloc[0])
            tele = load_car_data_sample(session_key)
            if tele.empty:
                st.info("car_data.parquet is missing or empty for this session.")
            else:
                col1, col2 = st.columns(2)
                raw_cols = existing(tele, ["Speed", "RPM", "Throttle", "Brake", "DRS", "nGear"])
                if raw_cols:
                    long = tele[raw_cols].melt(var_name="feature", value_name="value").dropna()
                    fig = px.histogram(long, x="value", color="feature", facet_col="feature", facet_col_wrap=3, nbins=60, title=f"Raw Telemetry Distributions - Session {session_key}")
                    fig.update_yaxes(matches=None)
                    fig.update_xaxes(matches=None)
                    col1.plotly_chart(fig, use_container_width=True)
                if {"Throttle", "Speed", "Brake"}.issubset(tele.columns):
                    plot = tele.dropna(subset=["Throttle", "Speed", "Brake"]).sample(min(len(tele.dropna(subset=["Throttle", "Speed", "Brake"])), 20_000), random_state=42)
                    fig = px.scatter(plot, x="Throttle", y="Speed", color="Brake", opacity=0.3, title="Throttle / Brake / Speed Relationship")
                    col2.plotly_chart(fig, use_container_width=True)

    elif chart == "Location spread" and "loc_prev_loc_spread_xy" in filtered.columns:
        fig = px.histogram(filtered.dropna(subset=["loc_prev_loc_spread_xy"]), x="loc_prev_loc_spread_xy", nbins=60)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Location feature relationship":
        col1, col2 = st.columns(2)
        if "loc_prev_loc_bbox_area" in filtered.columns:
            fig = px.histogram(filtered.dropna(subset=["loc_prev_loc_bbox_area"]), x="loc_prev_loc_bbox_area", nbins=60, title="Previous-Lap Location BBox Area")
            col1.plotly_chart(fig, use_container_width=True)
        if {"loc_prev_loc_bbox_area", "tele_prev_speed_max"}.issubset(sample.columns):
            fig = px.scatter(sample.dropna(subset=["loc_prev_loc_bbox_area", "tele_prev_speed_max"]), x="loc_prev_loc_bbox_area", y="tele_prev_speed_max", opacity=0.25, title="Location BBox Area vs Previous Max Speed")
            col2.plotly_chart(fig, use_container_width=True)

    elif chart == "Raw location sample":
        if "session_key" not in filtered.columns:
            st.info("session_key is missing.")
        else:
            session_key = int(filtered["session_key"].dropna().iloc[0])
            loc = load_location_sample(session_key)
            if loc.empty:
                st.info("location.parquet is missing or empty for this session.")
            else:
                col1, col2, col3 = st.columns(3)
                fig = px.scatter(loc, x="X", y="Y", color="driver_number" if "driver_number" in loc.columns else None, title=f"Track Map Scatter - Session {session_key}", opacity=0.45)
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                col1.plotly_chart(fig, use_container_width=True)
                if {"driver_number", "lap_number"}.issubset(loc.columns):
                    lap_counts = loc.groupby(["driver_number", "lap_number"]).size().reset_index(name="points")
                    fig = px.histogram(lap_counts, x="points", nbins=50, title="GPS Points Per Driver-Lap")
                    col2.plotly_chart(fig, use_container_width=True)
                    driver = loc["driver_number"].value_counts().index[0]
                    sample_laps = loc[loc["driver_number"] == driver]["lap_number"].dropna().value_counts().head(3).index.tolist()
                    tmp = loc[(loc["driver_number"] == driver) & (loc["lap_number"].isin(sample_laps))]
                    fig = px.scatter(tmp, x="X", y="Y", color="lap_number", title=f"Trajectory Comparison Driver {driver}", opacity=0.55)
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    col3.plotly_chart(fig, use_container_width=True)

    elif chart == "Overtakes" and {"overtakes_made_so_far", "overtakes_lost_so_far", "net_overtakes_so_far"}.issubset(filtered.columns):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=filtered["overtakes_made_so_far"], name="made", opacity=0.75))
        fig.add_trace(go.Histogram(x=filtered["overtakes_lost_so_far"], name="lost", opacity=0.75))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered[["overtakes_made_so_far", "overtakes_lost_so_far", "net_overtakes_so_far"]].describe().T, use_container_width=True)

    elif chart == "Overtakes vs target" and {"net_overtakes_so_far", "target_win"}.issubset(filtered.columns):
        tmp = filtered.dropna(subset=["net_overtakes_so_far", "target_win"]).copy()
        tmp["net_overtake_bin"] = pd.cut(tmp["net_overtakes_so_far"], bins=[-20, -5, -1, 0, 1, 5, 20], include_lowest=True)
        rates = tmp.groupby("net_overtake_bin", observed=True)["target_win"].mean().reset_index()
        rates["net_overtake_bin"] = rates["net_overtake_bin"].astype(str)
        fig = px.bar(rates, x="net_overtake_bin", y="target_win", title="Win Target Rate by Net Overtakes")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Team radio count":
        team_radio = load_clean_csv("team_radio")
        overtakes = load_clean_csv("overtakes")
        col1, col2, col3 = st.columns(3)
        if not overtakes.empty:
            driver_col = "overtaking_driver_number" if "overtaking_driver_number" in overtakes.columns else "driver_number"
            if driver_col in overtakes.columns:
                top = overtakes[driver_col].value_counts().head(15).reset_index()
                top.columns = ["driver", "overtakes"]
                fig = px.bar(top, y="driver", x="overtakes", orientation="h", title="Top Overtaking Drivers")
                col1.plotly_chart(fig, use_container_width=True)
            if "position" in overtakes.columns:
                fig = px.histogram(overtakes, x="position", nbins=30, title="Overtakes By Position")
                col2.plotly_chart(fig, use_container_width=True)
        if not team_radio.empty and {"session_key", "driver_number"}.issubset(team_radio.columns):
            radio_count = team_radio.groupby(["session_key", "driver_number"]).size().reset_index(name="radio_count")
            fig = px.histogram(radio_count, x="radio_count", nbins=40, title="Team Radio Count Per Driver-Session")
            col3.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Required columns for this chart are missing in the current dataset.")


def replay_tab(df: pd.DataFrame) -> None:
    st.subheader("Race Replay")
    st.caption(
        "Canvas replay built from cleaned FastF1 location coordinates. "
        "It follows the replay-loop pattern from f1-race-replay: build compact frames once, then update car positions in one canvas."
    )

    if not LOCATION_PATH.exists():
        st.warning(f"Missing location parquet: {LOCATION_PATH}")
        return
    if "session_key" not in df.columns:
        st.warning("Master dataset does not contain session_key.")
        return

    session_cols = existing(df, ["session_key", "year", "session_type", "circuit_short_name", "country_name"])
    sessions = (
        df[session_cols]
        .drop_duplicates("session_key")
        .sort_values([c for c in ["year", "session_type", "circuit_short_name", "session_key"] if c in session_cols])
        .copy()
    )
    if "session_type" in sessions.columns:
        sessions = sessions[sessions["session_type"].isin(["Race", "Sprint"])].copy()
    available_sessions = load_available_location_sessions()
    if available_sessions:
        sessions = sessions[sessions["session_key"].astype(int).isin(available_sessions)].copy()
    if sessions.empty:
        st.warning("No Race/Sprint sessions with location data are available for replay.")
        return

    sessions["label"] = sessions.apply(
        lambda row: " | ".join(
            str(row[col])
            for col in ["year", "session_type", "circuit_short_name", "country_name", "session_key"]
            if col in sessions.columns and pd.notna(row[col])
        ),
        axis=1,
    )
    label_to_session = dict(zip(sessions["label"], sessions["session_key"]))

    col1, col2, col3 = st.columns([2, 1, 1])
    selected_label = col1.selectbox("Session", sessions["label"].tolist(), index=0)
    frame_step = col2.select_slider("Frame step", options=[1, 2, 5, 10, 15, 30], value=2)
    max_frames = col3.select_slider("Max frames", options=[300, 600, 900, 1200, 1800], value=900)
    selected_session = int(label_to_session[selected_label])

    session_rows = df[df["session_key"] == selected_session].copy()
    driver_cols = existing(session_rows, ["driver_number", "full_name", "name_acronym", "team_name"])
    drivers = session_rows[driver_cols].drop_duplicates("driver_number").sort_values("driver_number")
    drivers["driver_number"] = pd.to_numeric(drivers["driver_number"], errors="coerce").astype("Int64")
    drivers = drivers.dropna(subset=["driver_number"]).copy()
    drivers["driver_number_int"] = drivers["driver_number"].astype(int)
    drivers["label"] = drivers.apply(
        lambda row: " | ".join(
            part
            for part in [
                f"#{row['driver_number_int']}",
                str(row.get("name_acronym", "") or "").strip(),
                str(row.get("full_name", "") or "").strip(),
                str(row.get("team_name", "") or "").strip(),
            ]
            if part and part != "nan"
        ),
        axis=1,
    )
    driver_label_to_number = dict(zip(drivers["label"], drivers["driver_number_int"]))
    default_driver_labels = drivers["label"].head(10).tolist()
    selected_driver_labels = st.multiselect(
        "Drivers to display",
        drivers["label"].tolist(),
        default=default_driver_labels,
        help="Replay only renders selected drivers.",
    )
    selected_drivers = tuple(driver_label_to_number[label] for label in selected_driver_labels)
    if not selected_drivers:
        st.warning("Select at least one driver to render the replay.")
        return

    driver_labels = {driver_label_to_number[label]: label for label in selected_driver_labels}
    replay_request = {
        "session": selected_session,
        "frame_step": frame_step,
        "max_frames": max_frames,
        "drivers": selected_drivers,
        "version": REPLAY_PAYLOAD_VERSION,
    }
    load_replay = st.button("Load replay", type="primary")
    if not load_replay and st.session_state.get("last_replay_request") != replay_request:
        st.info("Choose session/drivers, then click `Load replay` to build the replay.")
        return
    if load_replay:
        st.session_state["last_replay_request"] = replay_request

    with st.spinner("Building compact replay frames from location parquet..."):
        replay_payload = load_replay_payload(
            selected_session,
            frame_step,
            selected_drivers,
            driver_labels,
            max_frames,
            REPLAY_PAYLOAD_VERSION,
        )
    if not replay_payload.get("frames"):
        st.warning("No location rows available for this selection.")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Replay frames", f"{len(replay_payload['frames']):,}")
    col_b.metric("Drivers in replay", f"{len(replay_payload['drivers']):,}")
    col_c.metric("Track points", f"{len(replay_payload['track']):,}")
    st.caption(f"Replay payload version: `{REPLAY_PAYLOAD_VERSION}`")
    st.info(
        "Replay uses one canvas loop with selected drivers, a simple coordinate cloud for the track, a live speed chart, and a leaderboard."
    )

    template = REPLAY_TEMPLATE_PATH.read_text(encoding="utf-8")
    payload_json = json.dumps(replay_payload, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__REPLAY_PAYLOAD__", payload_json)
    html = html.replace("</body>", f"<!-- {REPLAY_PAYLOAD_VERSION} --></body>")
    components.html(html, height=820, scrolling=False)

    with st.expander("Replay frame sample"):
        rows = []
        for frame in replay_payload["frames"][:10]:
            for car in frame["cars"]:
                rows.append({"t": frame["t"], **car})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

def preview_tab(filtered: pd.DataFrame, metadata: dict) -> None:
    st.subheader("Data Preview")
    view_mode = st.radio("Columns", ["Core columns", "Model-safe features", "All columns"], horizontal=True)
    if view_mode == "Core columns":
        cols = existing(
            filtered,
            KEY_COLUMNS
            + [
                "year",
                "session_type",
                "circuit_short_name",
                "full_name",
                "team_name",
                "compound",
                "lap_duration",
                "target_win",
                "target_podium",
                "target_top10",
            ],
        )
    elif view_mode == "Model-safe features":
        cols = existing(filtered, metadata.get("model_feature_columns", []))
    else:
        cols = list(filtered.columns)
    st.dataframe(filtered[cols].head(1000), use_container_width=True)


def main() -> None:
    st.title("F1 Final Dataset Explorer")
    st.caption("Data-quality and feature-exploration dashboard for the final lap-level dataset.")

    df = load_master()
    metadata = load_metadata()
    audit = load_optional_csv(AUDIT_PATH)
    cleaning_summary = load_optional_csv(CLEANING_SUMMARY_PATH)

    if df.empty:
        st.error(f"Missing final dataset: {MASTER_PATH}")
        st.info("Run `python run_e2e_pipeline.py` from the project root to create the dataset.")
        return

    filtered = make_filters(df)

    tabs = st.tabs(["Overview", "Quality", "Feature Groups", "Live Contract", "EDA", "Race Replay", "Data Preview"])
    with tabs[0]:
        overview_tab(filtered, filtered, metadata)
    with tabs[1]:
        quality_tab(filtered, metadata, audit, cleaning_summary)
    with tabs[2]:
        feature_groups_tab(filtered, metadata)
    with tabs[3]:
        live_contract_tab(filtered, metadata)
    with tabs[4]:
        eda_tab(filtered)
    with tabs[5]:
        replay_tab(filtered)
    with tabs[6]:
        preview_tab(filtered, metadata)


if __name__ == "__main__":
    main()

"""Build Silver EDA notebooks for cleaned data verification."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "eda" / "silver" / "notebooks"


def md(text: str):
    return nbformat.v4.new_markdown_cell(text)


def code(text: str):
    return nbformat.v4.new_code_cell(text)


def write_notebook(cells: list, path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    nbformat.write(notebook, path)


SETUP = """from pathlib import Path
import sys
from datetime import datetime
import json

import pandas as pd
import numpy as np
import plotly.express as px
import pyarrow.parquet as pq

ROOT = Path.cwd()
while not (ROOT / "configs" / "pipeline_config.yaml").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

SHARED = ROOT / "eda" / "shared" / "scripts"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from config import CLEANED_DATA_PATH

NOTEBOOK_NAME = "01_verify_cleaned_data"
OUTPUT_TABLES = ROOT / "eda" / "silver" / "outputs" / "tables" / NOTEBOOK_NAME
OUTPUT_CHARTS = ROOT / "eda" / "silver" / "outputs" / "charts" / NOTEBOOK_NAME
OUTPUT_REPORTS = ROOT / "eda" / "silver" / "outputs" / "reports" / NOTEBOOK_NAME
INSIGHTS = ROOT / "eda" / "silver" / "insights"
CHECKPOINTS = ROOT / "eda" / "silver" / "checkpoints"
for path in [OUTPUT_TABLES, OUTPUT_CHARTS, OUTPUT_REPORTS, INSIGHTS, CHECKPOINTS]:
    path.mkdir(parents=True, exist_ok=True)

def write_report(name: str, payload: dict) -> None:
    (OUTPUT_REPORTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def write_insight(title: str, observations: list[str], issues: list[str], recommendations: list[str]) -> None:
    content = f"# {title}\\n\\n"
    content += f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
    content += "## Key Observations\\n\\n" + "\\n".join(f"- {item}" for item in observations) + "\\n\\n"
    content += "## Issues\\n\\n" + ("\\n".join(f"- {item}" for item in issues) if issues else "- None") + "\\n\\n"
    content += "## Recommendations\\n\\n" + "\\n".join(f"- {item}" for item in recommendations) + "\\n"
    (INSIGHTS / f"{NOTEBOOK_NAME}.md").write_text(content, encoding="utf-8")

print("=" * 72)
print(f"SILVER EDA - {NOTEBOOK_NAME}")
print(f"Start time: {datetime.now()}")
print(f"Cleaned data path: {CLEANED_DATA_PATH}")
print("=" * 72)
"""


def build() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook([
        md("""# 01 Verify Cleaned Data

This notebook verifies that the Silver layer is usable for feature engineering after applying the Bronze-informed cleaning policy.

The notebook intentionally avoids loading full telemetry tables into memory. Telemetry row counts and schemas come from Parquet metadata, while detailed profiling is limited to operational tables."""),
        code(SETUP),
        md("""## File Coverage

Silver should contain one Parquet artifact per active endpoint. This check verifies file presence, row counts, and storage footprint without treating telemetry file size as a quality signal."""),
        code("""expected_files = [
    "meetings.parquet",
    "sessions.parquet",
    "drivers.parquet",
    "session_result.parquet",
    "laps.parquet",
    "weather.parquet",
    "stints.parquet",
    "starting_grid.parquet",
    "intervals.parquet",
    "position.parquet",
    "overtakes.parquet",
    "pit.parquet",
    "car_data.parquet",
    "location.parquet",
]
records = []
for name in expected_files:
    path = CLEANED_DATA_PATH / name
    if not path.exists():
        records.append({"file": name, "exists": False, "rows": 0, "columns": 0, "size_mb": 0.0, "endpoint_group": "Missing"})
        continue
    parquet_file = pq.ParquetFile(path)
    records.append({
        "file": name,
        "exists": True,
        "rows": int(parquet_file.metadata.num_rows),
        "columns": len(parquet_file.schema.names),
        "size_mb": path.stat().st_size / 1_000_000,
        "endpoint_group": "Telemetry" if name in {"car_data.parquet", "location.parquet"} else "Operational",
    })
file_df = pd.DataFrame(records)
file_df.to_csv(OUTPUT_TABLES / "file_inventory.csv", index=False)
display(file_df)"""),
        code("""fig = px.bar(
    file_df,
    x="file",
    y="rows",
    color="endpoint_group",
    log_y=True,
    title="Silver File Coverage: Row Counts by Endpoint",
    labels={"rows": "Rows (log scale)", "file": "Silver artifact"},
)
fig.update_xaxes(tickangle=35)
fig.write_html(OUTPUT_CHARTS / "file_row_coverage.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Race and Sprint Scope

OpenF1 encodes Sprint race rows with `session_type = Race` and `session_name = Sprint`. Silver therefore creates a normalized `event_type` so feature engineering can distinguish Grand Prix races from Sprint races."""),
        code("""sessions = pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet")
sessions["event_type"] = np.where(
    sessions["session_name"].astype(str).str.lower().eq("sprint"),
    "SPRINT_RACE",
    "GRAND_PRIX_RACE",
)
session_distribution = (
    sessions.groupby(["year", "event_type"], dropna=False)
    .size()
    .reset_index(name="sessions")
    .sort_values(["year", "event_type"])
)
session_distribution.to_csv(OUTPUT_TABLES / "session_distribution.csv", index=False)
display(session_distribution)"""),
        code("""fig = px.bar(
    session_distribution,
    x="year",
    y="sessions",
    color="event_type",
    barmode="group",
    title="Silver Session Scope: Grand Prix Race vs Sprint Race",
)
fig.write_html(OUTPUT_CHARTS / "session_distribution.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Critical Null Review

This check focuses on operational tables used immediately in Gold. Structural nulls from Bronze are allowed, but technical keys and core ML joins must remain populated."""),
        code("""critical_columns = {
    "sessions": ["session_key", "meeting_key", "session_name"],
    "drivers": ["session_key", "driver_number", "full_name"],
    "session_result": ["session_key", "driver_number", "position"],
    "laps": ["session_key", "driver_number", "lap_number", "lap_duration"],
    "weather": ["session_key", "date", "track_temperature", "air_temperature"],
    "starting_grid": ["session_key", "driver_number", "position"],
}
null_records = []
for table, columns in critical_columns.items():
    df = pd.read_parquet(CLEANED_DATA_PATH / f"{table}.parquet")
    for column in columns:
        if column not in df.columns:
            null_records.append({"table": table, "column": column, "rows": len(df), "null_count": len(df), "null_pct": 100.0, "status": "FAIL_MISSING_COLUMN"})
            continue
        null_count = int(df[column].isna().sum())
        null_records.append({
            "table": table,
            "column": column,
            "rows": len(df),
            "null_count": null_count,
            "null_pct": null_count / len(df) * 100 if len(df) else 0.0,
            "status": "PASS" if null_count == 0 else "REVIEW",
        })
null_df = pd.DataFrame(null_records)
null_df.to_csv(OUTPUT_TABLES / "critical_null_review.csv", index=False)
display(null_df.sort_values(["status", "null_pct"], ascending=[True, False]))"""),
        code("""plot_df = null_df[null_df["null_count"] > 0]
fig = px.bar(
    plot_df,
    x="null_pct",
    y="table",
    color="column",
    orientation="h",
    title="Silver Critical Null Review",
)
fig.write_html(OUTPUT_CHARTS / "critical_null_review.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Predictive Readiness Signals

The next checks validate the main race modeling signals: finish position distribution, lap-duration behavior, weather context, and grid-to-finish relationship."""),
        code("""session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
laps = pd.read_parquet(CLEANED_DATA_PATH / "laps.parquet")
weather = pd.read_parquet(CLEANED_DATA_PATH / "weather.parquet")
starting_grid = pd.read_parquet(CLEANED_DATA_PATH / "starting_grid.parquet")

finish = session_result.copy()
finish["finish_position"] = pd.to_numeric(finish["position"], errors="coerce")
finish_distribution = finish["finish_position"].value_counts(dropna=False).sort_index().reset_index()
finish_distribution.columns = ["finish_position", "drivers"]
finish_distribution.to_csv(OUTPUT_TABLES / "finish_position_distribution.csv", index=False)
display(finish_distribution.head(25))"""),
        code("""fig = px.bar(
    finish_distribution,
    x="finish_position",
    y="drivers",
    title="Finish Position Distribution",
    labels={"finish_position": "Finish position", "drivers": "Driver-session rows"},
)
fig.write_html(OUTPUT_CHARTS / "finish_position_distribution.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""lap_stats = laps["lap_duration"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).reset_index()
lap_stats.columns = ["metric", "lap_duration_seconds"]
lap_stats.to_csv(OUTPUT_TABLES / "lap_duration_stats.csv", index=False)
display(lap_stats)"""),
        code("""sample_laps = laps[["lap_duration"]].dropna()
if len(sample_laps) > 100_000:
    sample_laps = sample_laps.sample(100_000, random_state=42)
fig = px.histogram(sample_laps, x="lap_duration", nbins=80, title="Lap Duration Distribution After Silver Cleaning")
fig.write_html(OUTPUT_CHARTS / "lap_duration_distribution.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""weather_cols = [column for column in ["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed"] if column in weather.columns]
weather_stats = weather[weather_cols].describe().round(2).reset_index().rename(columns={"index": "metric"})
weather_stats.to_csv(OUTPUT_TABLES / "weather_stats.csv", index=False)
display(weather_stats)"""),
        code("""grid_finish = starting_grid.merge(
    session_result[["session_key", "driver_number", "position"]],
    on=["session_key", "driver_number"],
    suffixes=("_grid", "_finish"),
)
grid_finish["grid_position"] = pd.to_numeric(grid_finish["position_grid"], errors="coerce")
grid_finish["finish_position"] = pd.to_numeric(grid_finish["position_finish"], errors="coerce")
grid_finish = grid_finish.dropna(subset=["grid_position", "finish_position"])
grid_finish.to_csv(OUTPUT_TABLES / "grid_finish_sample.csv", index=False)
corr = float(grid_finish[["grid_position", "finish_position"]].corr().iloc[0, 1]) if len(grid_finish) > 1 else float("nan")
display(grid_finish[["session_key", "driver_number", "grid_position", "finish_position"]].head(20))"""),
        code("""fig = px.scatter(
    grid_finish,
    x="grid_position",
    y="finish_position",
    title=f"Grid Position vs Finish Position (corr={corr:.3f})",
)
if len(grid_finish) > 2:
    fit = np.polyfit(grid_finish["grid_position"], grid_finish["finish_position"], deg=1)
    x_line = np.array([grid_finish["grid_position"].min(), grid_finish["grid_position"].max()])
    y_line = fit[0] * x_line + fit[1]
    fig.add_scatter(x=x_line, y=y_line, mode="lines", name="Linear fit")
fig.write_html(OUTPUT_CHARTS / "grid_vs_finish.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Final Silver Gate

The Silver gate passes when all expected files exist, no critical columns are missing, and cleaned sessions contain only Grand Prix race or Sprint race rows."""),
        code("""missing_files = file_df[~file_df["exists"]]
missing_columns = null_df[null_df["status"].eq("FAIL_MISSING_COLUMN")]
invalid_event_rows = sessions[~sessions["event_type"].isin(["GRAND_PRIX_RACE", "SPRINT_RACE"])]
review_nulls = null_df[null_df["status"].eq("REVIEW")]

status = "PASS" if missing_files.empty and missing_columns.empty and invalid_event_rows.empty else "FAIL"
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "status": status,
    "total_files": int(len(file_df)),
    "total_rows": int(file_df["rows"].sum()),
    "grand_prix_sessions": int((sessions["event_type"] == "GRAND_PRIX_RACE").sum()),
    "sprint_sessions": int((sessions["event_type"] == "SPRINT_RACE").sum()),
    "critical_null_review_count": int(len(review_nulls)),
    "grid_finish_correlation": corr,
    "missing_files": missing_files.to_dict("records"),
    "missing_columns": missing_columns.to_dict("records"),
}
write_report("verify_cleaned_data", report)
write_insight(
    "Silver Verification Insights",
    [
        f"Verified {len(file_df)} Silver Parquet artifacts.",
        f"Grand Prix races: {report['grand_prix_sessions']}; Sprint races: {report['sprint_sessions']}.",
        f"Grid-to-finish correlation: {corr:.3f}.",
    ],
    [f"{row.table}.{row.column}: {row.null_count} nulls" for row in review_nulls.itertuples()],
    [
        "Use event_type rather than session_type to distinguish Grand Prix races from Sprint races.",
        "Treat remaining critical null reviews as feature-level decisions, not file integrity failures.",
        "Proceed to Gold only after this Silver verification status remains PASS.",
    ],
)
if status == "PASS":
    (CHECKPOINTS / "silver_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "01_verify_cleaned_data.ipynb")

    driver_setup = SETUP.replace('NOTEBOOK_NAME = "01_verify_cleaned_data"', 'NOTEBOOK_NAME = "02_driver_analysis"')
    write_notebook([
        md("""# 02 Driver Analysis

This notebook analyzes driver-level competitive patterns in the Silver layer: win concentration, lap-time consistency, overtaking balance, grid-to-finish dynamics, teammate gaps, and year-over-year performance trends.

The analysis uses only operational Silver tables and avoids telemetry-scale data."""),
        code(driver_setup),
        code("""drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
laps = pd.read_parquet(CLEANED_DATA_PATH / "laps.parquet")
overtakes = pd.read_parquet(CLEANED_DATA_PATH / "overtakes.parquet")
starting_grid = pd.read_parquet(CLEANED_DATA_PATH / "starting_grid.parquet")
sessions = pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet")

sessions["event_type"] = np.where(
    sessions["session_name"].astype(str).str.lower().eq("sprint"),
    "SPRINT_RACE",
    "GRAND_PRIX_RACE",
)
driver_dim = (
    drivers[["session_key", "driver_number", "full_name", "team_name"]]
    .drop_duplicates(["session_key", "driver_number"])
)
results = (
    session_result
    .merge(sessions[["session_key", "year", "event_type"]], on="session_key", how="left")
    .merge(driver_dim, on=["session_key", "driver_number"], how="left")
)
results["driver_id"] = results["full_name"].fillna("Driver " + results["driver_number"].astype(str))
results["finish_pos"] = pd.to_numeric(results["position"], errors="coerce")
results["is_classified"] = results["finish_pos"].notna()
print(f"Drivers: {results['driver_id'].nunique()}")
print(f"Sessions: {sessions['session_key'].nunique()} ({sessions['event_type'].value_counts().to_dict()})")
print(f"Result rows: {len(results):,}")
print(f"Laps: {len(laps):,}")
print(f"Overtakes: {len(overtakes):,}")"""),
        md("""## 1. Win Rate and Competitive Concentration

Formula 1 wins are usually concentrated among a small set of drivers. This section quantifies how concentrated wins are in the cleaned race and sprint sample."""),
        code("""driver_summary = (
    results.groupby("driver_id", as_index=False)
    .agg(
        starts=("session_key", "nunique"),
        classified_finishes=("is_classified", "sum"),
        wins=("finish_pos", lambda s: int((s == 1).sum())),
        podiums=("finish_pos", lambda s: int((s <= 3).sum())),
        top10s=("finish_pos", lambda s: int((s <= 10).sum())),
        avg_finish=("finish_pos", "mean"),
        points=("points", "sum"),
        primary_driver_number=("driver_number", lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(s.iloc[0])),
        primary_team=("team_name", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Unknown"),
    )
)
driver_summary["win_rate_pct"] = np.where(driver_summary["starts"] > 0, driver_summary["wins"] / driver_summary["starts"] * 100, 0)
driver_summary["podium_rate_pct"] = np.where(driver_summary["starts"] > 0, driver_summary["podiums"] / driver_summary["starts"] * 100, 0)
driver_summary["top10_rate_pct"] = np.where(driver_summary["starts"] > 0, driver_summary["top10s"] / driver_summary["starts"] * 100, 0)
driver_summary = driver_summary.sort_values(["wins", "points", "avg_finish"], ascending=[False, False, True])
driver_summary.to_csv(OUTPUT_TABLES / "driver_result_summary.csv", index=False)
display(driver_summary.head(20))"""),
        code("""qualified = driver_summary[driver_summary["starts"] >= 5].copy()
win_rates = qualified["win_rate_pct"]
total_wins = float(driver_summary["wins"].sum())
top1_pct = float(driver_summary.head(1)["wins"].sum() / total_wins * 100) if total_wins else 0.0
top3_pct = float(driver_summary.head(3)["wins"].sum() / total_wins * 100) if total_wins else 0.0
top5_pct = float(driver_summary.head(5)["wins"].sum() / total_wins * 100) if total_wins else 0.0

fig = px.bar(
    driver_summary.head(20).sort_values("wins"),
    x="wins",
    y="driver_id",
    orientation="h",
    color="win_rate_pct",
    title="Top Drivers by Wins and Win Rate",
    labels={"driver_id": "Driver", "wins": "Wins", "win_rate_pct": "Win rate %"},
)
fig.write_html(OUTPUT_CHARTS / "driver_wins.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""lorenz = driver_summary.sort_values("wins", ascending=True).copy()
lorenz["cumulative_wins_pct"] = lorenz["wins"].cumsum() / total_wins * 100 if total_wins else 0
lorenz["drivers_pct"] = (np.arange(len(lorenz)) + 1) / len(lorenz) * 100 if len(lorenz) else []
lorenz.to_csv(OUTPUT_TABLES / "win_concentration_lorenz.csv", index=False)
fig = px.line(lorenz, x="drivers_pct", y="cumulative_wins_pct", title="Win Concentration Lorenz Curve")
fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Equal distribution")
fig.write_html(OUTPUT_CHARTS / "win_concentration.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 2. Lap-Time Consistency vs Finishing Outcome

A stable driver profile should show lower lap-time variance and stronger average finishing outcomes. This check aggregates completed laps by driver and compares consistency with average finish position."""),
        code("""lap_base = laps[laps["lap_duration"].notna()].copy()
lap_base["lap_duration"] = pd.to_numeric(lap_base["lap_duration"], errors="coerce")
lap_base = lap_base[lap_base["lap_duration"].between(50, 900)]
lap_base = lap_base.merge(driver_dim, on=["session_key", "driver_number"], how="left")
lap_base["driver_id"] = lap_base["full_name"].fillna("Driver " + lap_base["driver_number"].astype(str))
consistency = (
    lap_base.groupby("driver_id", as_index=False)
    .agg(
        avg_lap_time=("lap_duration", "mean"),
        std_lap_time=("lap_duration", "std"),
        median_lap_time=("lap_duration", "median"),
        lap_count=("lap_duration", "count"),
        outlier_laps=("is_outlier_lap", "sum"),
        primary_driver_number=("driver_number", lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(s.iloc[0])),
        primary_team=("team_name", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Unknown"),
    )
)
avg_finish = results.groupby("driver_id", as_index=False).agg(avg_finish=("finish_pos", "mean"), starts=("session_key", "nunique"))
consistency = consistency.merge(avg_finish, on="driver_id", how="left")
consistency = consistency[(consistency["lap_count"] >= 100) & consistency["avg_finish"].notna()]
consistency_corr = float(consistency["std_lap_time"].corr(consistency["avg_finish"])) if len(consistency) > 1 else float("nan")
consistency.to_csv(OUTPUT_TABLES / "driver_lap_consistency.csv", index=False)
display(consistency.sort_values("std_lap_time").head(20))"""),
        code("""fig = px.scatter(
    consistency,
    x="std_lap_time",
    y="avg_finish",
    size="lap_count",
    hover_name="driver_id",
    title=f"Lap-Time Consistency vs Average Finish (corr={consistency_corr:.3f})",
    labels={"std_lap_time": "Lap duration standard deviation", "avg_finish": "Average finish position"},
)
if len(consistency) > 2:
    fit = np.polyfit(consistency["std_lap_time"], consistency["avg_finish"], deg=1)
    x_line = np.array([consistency["std_lap_time"].min(), consistency["std_lap_time"].max()])
    fig.add_scatter(x=x_line, y=fit[0] * x_line + fit[1], mode="lines", name="Linear fit")
fig.update_yaxes(autorange="reversed")
fig.write_html(OUTPUT_CHARTS / "consistency_vs_finish.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 3. Overtaking Balance

Overtake balance measures racecraft and traffic exposure. A positive net value means the driver made more overtakes than they received."""),
        code("""overtaker_dim = driver_dim.rename(columns={"driver_number": "overtaking_driver_number", "full_name": "overtaking_driver_id"})
overtaken_dim = driver_dim.rename(columns={"driver_number": "overtaken_driver_number", "full_name": "overtaken_driver_id"})
overtakes_named = (
    overtakes
    .merge(overtaker_dim[["session_key", "overtaking_driver_number", "overtaking_driver_id"]], on=["session_key", "overtaking_driver_number"], how="left")
    .merge(overtaken_dim[["session_key", "overtaken_driver_number", "overtaken_driver_id"]], on=["session_key", "overtaken_driver_number"], how="left")
)
overtakes_named["overtaking_driver_id"] = overtakes_named["overtaking_driver_id"].fillna("Driver " + overtakes_named["overtaking_driver_number"].astype(str))
overtakes_named["overtaken_driver_id"] = overtakes_named["overtaken_driver_id"].fillna("Driver " + overtakes_named["overtaken_driver_number"].astype(str))
made = overtakes_named.groupby("overtaking_driver_id").size().reset_index(name="overtakes_made").rename(columns={"overtaking_driver_id": "driver_id"})
received = overtakes_named.groupby("overtaken_driver_id").size().reset_index(name="overtakes_received").rename(columns={"overtaken_driver_id": "driver_id"})
overtake_stats = made.merge(received, on="driver_id", how="outer").fillna(0)
overtake_stats["net_overtakes"] = overtake_stats["overtakes_made"] - overtake_stats["overtakes_received"]
overtake_stats["overtake_ratio"] = overtake_stats["overtakes_made"] / (overtake_stats["overtakes_received"] + 1)
overtake_stats = overtake_stats.sort_values("net_overtakes", ascending=False)
overtake_stats.to_csv(OUTPUT_TABLES / "driver_overtake_balance.csv", index=False)
display(overtake_stats.head(20))"""),
        code("""plot_overtakes = overtake_stats.head(20).sort_values("net_overtakes")
fig = px.bar(
    plot_overtakes,
    x="net_overtakes",
    y="driver_id",
    orientation="h",
    color="net_overtakes",
    title="Top Net Overtake Balance by Driver",
)
fig.write_html(OUTPUT_CHARTS / "net_overtakes.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""fig = px.scatter(
    overtake_stats,
    x="overtakes_made",
    y="overtakes_received",
    hover_name="driver_id",
    color="net_overtakes",
    title="Overtakes Made vs Received",
)
max_axis = max(overtake_stats["overtakes_made"].max(), overtake_stats["overtakes_received"].max())
fig.add_scatter(x=[0, max_axis], y=[0, max_axis], mode="lines", name="Equal balance")
fig.write_html(OUTPUT_CHARTS / "overtakes_made_vs_received.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 4. Grid-to-Finish Dynamics

Starting position is one of the strongest pre-race predictors. This section quantifies position change and how often wins come from different grid slots."""),
        code("""grid_finish = starting_grid.merge(
    session_result[["session_key", "driver_number", "position"]],
    on=["session_key", "driver_number"],
    suffixes=("_grid", "_finish"),
)
grid_finish = grid_finish.merge(driver_dim, on=["session_key", "driver_number"], how="left")
grid_finish["driver_id"] = grid_finish["full_name"].fillna("Driver " + grid_finish["driver_number"].astype(str))
grid_finish["grid_pos"] = pd.to_numeric(grid_finish["position_grid"], errors="coerce")
grid_finish["finish_pos"] = pd.to_numeric(grid_finish["position_finish"], errors="coerce")
grid_finish = grid_finish.dropna(subset=["grid_pos", "finish_pos"])
grid_finish["position_change"] = grid_finish["grid_pos"] - grid_finish["finish_pos"]
grid_corr = float(grid_finish["grid_pos"].corr(grid_finish["finish_pos"])) if len(grid_finish) > 1 else float("nan")
grid_finish.to_csv(OUTPUT_TABLES / "grid_finish_dynamics.csv", index=False)
display(grid_finish.sort_values("position_change", ascending=False).head(20))"""),
        code("""grid_finish["grid_bracket"] = pd.cut(
    grid_finish["grid_pos"],
    bins=[0, 3, 6, 10, 15, 25],
    labels=["P1-P3", "P4-P6", "P7-P10", "P11-P15", "P16+"],
)
bracket_stats = grid_finish.groupby("grid_bracket", observed=False).agg(
    avg_position_change=("position_change", "mean"),
    median_position_change=("position_change", "median"),
    drivers=("driver_number", "count"),
).reset_index()
bracket_stats.to_csv(OUTPUT_TABLES / "grid_bracket_position_change.csv", index=False)
display(bracket_stats)"""),
        code("""fig = px.scatter(
    grid_finish,
    x="grid_pos",
    y="finish_pos",
    color="position_change",
    hover_name="driver_id",
    title=f"Grid Position vs Finish Position (corr={grid_corr:.3f})",
)
fig.update_yaxes(autorange="reversed")
fig.write_html(OUTPUT_CHARTS / "grid_vs_finish_driver.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""fig = px.bar(
    bracket_stats,
    x="grid_bracket",
    y="avg_position_change",
    title="Average Position Change by Starting Bracket",
    labels={"avg_position_change": "Average positions gained", "grid_bracket": "Grid bracket"},
)
fig.write_html(OUTPUT_CHARTS / "grid_bracket_position_change.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 5. Teammate Gaps

Teammate comparisons control partially for car performance. Smaller intra-team gaps indicate a tighter pairing, while large gaps suggest driver-level or race-execution separation."""),
        code("""driver_team = drivers[["session_key", "driver_number", "full_name", "team_name"]].drop_duplicates(["session_key", "driver_number"])
team_results = session_result.merge(driver_team, on=["session_key", "driver_number"], how="left")
team_results["finish_pos"] = pd.to_numeric(team_results["position"], errors="coerce")
team_results = team_results.dropna(subset=["finish_pos", "team_name"])
team_gap_rows = []
for (session_key, team_name), group in team_results.groupby(["session_key", "team_name"]):
    if len(group) == 2:
        ordered = group.sort_values("finish_pos")
        team_gap_rows.append({
            "session_key": session_key,
            "team_name": team_name,
            "best_driver_number": int(ordered.iloc[0]["driver_number"]),
            "best_driver": ordered.iloc[0]["full_name"],
            "second_driver_number": int(ordered.iloc[1]["driver_number"]),
            "second_driver": ordered.iloc[1]["full_name"],
            "gap_positions": float(ordered.iloc[1]["finish_pos"] - ordered.iloc[0]["finish_pos"]),
        })
team_gaps = pd.DataFrame(team_gap_rows)
team_gap_summary = team_gaps.groupby("team_name", as_index=False).agg(
    avg_gap=("gap_positions", "mean"),
    median_gap=("gap_positions", "median"),
    paired_sessions=("session_key", "count"),
).sort_values("avg_gap")
team_gaps.to_csv(OUTPUT_TABLES / "teammate_gap_events.csv", index=False)
team_gap_summary.to_csv(OUTPUT_TABLES / "teammate_gap_summary.csv", index=False)
display(team_gap_summary)"""),
        code("""fig = px.bar(
    team_gap_summary,
    x="team_name",
    y="avg_gap",
    color="paired_sessions",
    title="Average Intra-Team Finish Gap",
    labels={"avg_gap": "Average finish-position gap", "team_name": "Team"},
)
fig.update_xaxes(tickangle=35)
fig.write_html(OUTPUT_CHARTS / "teammate_gap_summary.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 6. Season Trends

This section tracks whether driver outcomes improve or decline across the available 2024-2026 sample."""),
        code("""driver_year = (
    results.dropna(subset=["finish_pos"])
    .groupby(["driver_id", "year"], as_index=False)
    .agg(
        avg_finish=("finish_pos", "mean"),
        points=("points", "sum"),
        starts=("session_key", "nunique"),
        wins=("finish_pos", lambda s: int((s == 1).sum())),
        primary_driver_number=("driver_number", lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(s.iloc[0])),
        primary_team=("team_name", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "Unknown"),
    )
)
driver_year = driver_year[driver_year["starts"] >= 3].sort_values(["driver_id", "year"])
driver_year["prev_avg_finish"] = driver_year.groupby("driver_id")["avg_finish"].shift(1)
driver_year["finish_improvement"] = driver_year["prev_avg_finish"] - driver_year["avg_finish"]
driver_year.to_csv(OUTPUT_TABLES / "driver_year_trends.csv", index=False)
display(driver_year.head(30))"""),
        code("""season_avg = driver_year.groupby("year", as_index=False).agg(avg_finish=("avg_finish", "mean"), avg_points=("points", "mean"))
fig = px.line(season_avg, x="year", y="avg_finish", markers=True, title="Average Driver Finish by Season")
fig.update_yaxes(autorange="reversed")
fig.write_html(OUTPUT_CHARTS / "season_average_finish.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""improvement = driver_year.dropna(subset=["finish_improvement"]).copy()
fig = px.histogram(
    improvement,
    x="finish_improvement",
    nbins=25,
    title="Year-over-Year Finish Position Improvement",
    labels={"finish_improvement": "Improvement in average finish position"},
)
fig.add_vline(x=0, line_dash="dash")
fig.write_html(OUTPUT_CHARTS / "driver_improvement_distribution.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Final Driver Analysis Report

This report captures the headline driver-level signals available for Gold feature engineering."""),
        code("""report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "total_drivers": int(results["driver_id"].nunique()),
    "sessions_analyzed": int(sessions["session_key"].nunique()),
    "grand_prix_sessions": int((sessions["event_type"] == "GRAND_PRIX_RACE").sum()),
    "sprint_sessions": int((sessions["event_type"] == "SPRINT_RACE").sum()),
    "mean_win_rate_pct_min_5_starts": float(win_rates.mean()) if len(win_rates) else 0.0,
    "top3_win_concentration_pct": top3_pct,
    "lap_consistency_finish_corr": consistency_corr,
    "total_overtakes": int(len(overtakes)),
    "grid_finish_corr": grid_corr,
    "avg_teammate_gap": float(team_gaps["gap_positions"].mean()) if not team_gaps.empty else 0.0,
    "driver_year_rows": int(len(driver_year)),
}
write_report("driver_analysis", report)
write_insight(
    "Silver Driver Analysis Insights",
    [
        f"Analyzed {report['total_drivers']} drivers across {report['sessions_analyzed']} race/sprint sessions.",
        f"Top three drivers account for {report['top3_win_concentration_pct']:.1f}% of wins.",
        f"Grid-to-finish correlation is {report['grid_finish_corr']:.3f}.",
        f"Lap consistency to finish correlation is {report['lap_consistency_finish_corr']:.3f}.",
    ],
    [],
    [
        "Use driver_summary, consistency, overtake balance, and grid dynamics as Gold feature candidates.",
        "Use event_type to separate Grand Prix race and Sprint race behavior.",
        "Treat teammate gaps as contextual features because team pairing controls partially for car performance.",
    ],
)
(CHECKPOINTS / "silver_driver_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "02_driver_analysis.ipynb")

    team_setup = SETUP.replace('NOTEBOOK_NAME = "01_verify_cleaned_data"', 'NOTEBOOK_NAME = "03_team_analysis"')
    write_notebook([
        md("""# 03 Team Analysis

This notebook analyzes constructor-level performance patterns in the Silver layer: points concentration, competitive tiers, season trends, home-country effects, and intra-team balance.

The goal is not to praise one team, but to understand the team-level structure that Gold features should preserve."""),
        code(team_setup),
        code("""session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
sessions = pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet")
meetings = pd.read_parquet(CLEANED_DATA_PATH / "meetings.parquet")

sessions["event_type"] = np.where(
    sessions["session_name"].astype(str).str.lower().eq("sprint"),
    "SPRINT_RACE",
    "GRAND_PRIX_RACE",
)
driver_dim = drivers[["session_key", "driver_number", "full_name", "team_name"]].drop_duplicates(["session_key", "driver_number"])
team_results = (
    session_result
    .merge(driver_dim, on=["session_key", "driver_number"], how="left")
    .merge(sessions[["session_key", "year", "event_type", "meeting_key", "country_name", "circuit_short_name"]], on="session_key", how="left", suffixes=("", "_session"))
)
team_results["finish_pos"] = pd.to_numeric(team_results["position"], errors="coerce")
team_results["driver_id"] = team_results["full_name"].fillna("Driver " + team_results["driver_number"].astype(str))
team_results["team_name"] = team_results["team_name"].fillna("Unknown")
team_results["points"] = pd.to_numeric(team_results["points"], errors="coerce").fillna(0)
team_results = team_results.dropna(subset=["session_key", "team_name"])

print(f"Team-driver result rows: {len(team_results):,}")
print(f"Unique teams: {team_results['team_name'].nunique()}")
print(f"Race/Sprint sessions: {team_results['session_key'].nunique()} ({sessions['event_type'].value_counts().to_dict()})")
print(f"Countries: {team_results['country_name'].nunique()}")"""),
        md("""## 1. Points, Wins, and Podium Concentration

Team points are often more concentrated than driver starts. This section measures how much of the available performance is captured by the leading constructors."""),
        code("""team_session = (
    team_results.groupby(["session_key", "team_name"], as_index=False)
    .agg(
        team_points=("points", "sum"),
        best_finish=("finish_pos", "min"),
        classified_drivers=("finish_pos", lambda s: int(s.notna().sum())),
        drivers_entered=("driver_number", "nunique"),
        year=("year", "first"),
        event_type=("event_type", "first"),
        country_name=("country_name", "first"),
        circuit_short_name=("circuit_short_name", "first"),
    )
)
team_total = (
    team_session.groupby("team_name", as_index=False)
    .agg(
        total_points=("team_points", "sum"),
        sessions_entered=("session_key", "nunique"),
        avg_points_per_session=("team_points", "mean"),
        avg_best_finish=("best_finish", "mean"),
        wins=("best_finish", lambda s: int((s == 1).sum())),
        podium_sessions=("best_finish", lambda s: int((s <= 3).sum())),
    )
)
team_total["podium_session_rate_pct"] = team_total["podium_sessions"] / team_total["sessions_entered"] * 100
team_total["win_session_rate_pct"] = team_total["wins"] / team_total["sessions_entered"] * 100
team_total = team_total.sort_values(["total_points", "wins", "avg_best_finish"], ascending=[False, False, True])
total_points_all = float(team_total["total_points"].sum())
top1_points_pct = float(team_total.head(1)["total_points"].sum() / total_points_all * 100) if total_points_all else 0.0
top3_points_pct = float(team_total.head(3)["total_points"].sum() / total_points_all * 100) if total_points_all else 0.0
team_total.to_csv(OUTPUT_TABLES / "team_points_summary.csv", index=False)
display(team_total)"""),
        code("""fig = px.bar(
    team_total.sort_values("total_points"),
    x="total_points",
    y="team_name",
    orientation="h",
    color="podium_session_rate_pct",
    title="Team Points and Podium Rate",
    labels={"team_name": "Team", "total_points": "Total points", "podium_session_rate_pct": "Podium session rate %"},
)
fig.write_html(OUTPUT_CHARTS / "team_points_summary.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""concentration = team_total.sort_values("total_points", ascending=True).copy()
concentration["cumulative_points_pct"] = concentration["total_points"].cumsum() / total_points_all * 100 if total_points_all else 0
concentration["teams_pct"] = (np.arange(len(concentration)) + 1) / len(concentration) * 100
concentration.to_csv(OUTPUT_TABLES / "team_points_concentration.csv", index=False)
fig = px.line(concentration, x="teams_pct", y="cumulative_points_pct", title="Team Points Concentration")
fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Equal distribution")
fig.write_html(OUTPUT_CHARTS / "team_points_concentration.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 2. Competitive Hierarchy

Rather than hard-coding team labels, this section assigns tiers from the observed point distribution. The exact labels are descriptive and should be treated as EDA categories, not permanent business rules."""),
        code("""ranked = team_total.copy()
ranked["points_rank_pct"] = ranked["total_points"].rank(pct=True, method="first")
ranked["tier"] = pd.cut(
    ranked["points_rank_pct"],
    bins=[0, 1/3, 2/3, 1.0],
    labels=["Backmarker", "Midfield", "Top"],
    include_lowest=True,
)
tier_stats = ranked.groupby("tier", observed=False).agg(
    teams=("team_name", "count"),
    avg_points=("total_points", "mean"),
    avg_podium_rate=("podium_session_rate_pct", "mean"),
    avg_wins=("wins", "mean"),
).reset_index()
ranked.to_csv(OUTPUT_TABLES / "team_tiers.csv", index=False)
tier_stats.to_csv(OUTPUT_TABLES / "team_tier_stats.csv", index=False)
display(ranked[["team_name", "total_points", "wins", "podium_session_rate_pct", "tier"]])
display(tier_stats)"""),
        code("""fig = px.bar(
    ranked.sort_values("total_points"),
    x="total_points",
    y="team_name",
    color="tier",
    orientation="h",
    title="Observed Team Competitive Tiers",
)
fig.write_html(OUTPUT_CHARTS / "team_tiers.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 3. Team Performance Over Seasons

A team feature should capture both level and trajectory. This section tracks season points, average finish, and year-over-year point changes."""),
        code("""team_season = (
    team_results.groupby(["team_name", "year"], as_index=False)
    .agg(
        season_points=("points", "sum"),
        avg_finish=("finish_pos", "mean"),
        sessions=("session_key", "nunique"),
        wins=("finish_pos", lambda s: int((s == 1).sum())),
        podiums=("finish_pos", lambda s: int((s <= 3).sum())),
    )
)
team_season = team_season[team_season["sessions"] >= 3].sort_values(["team_name", "year"])
team_season["points_change_pct"] = team_season.groupby("team_name")["season_points"].pct_change() * 100
team_season["finish_improvement"] = team_season.groupby("team_name")["avg_finish"].shift(1) - team_season["avg_finish"]
team_season.to_csv(OUTPUT_TABLES / "team_season_trends.csv", index=False)
display(team_season.head(30))"""),
        code("""top_teams = team_total.head(8)["team_name"].tolist()
trend_plot = team_season[team_season["team_name"].isin(top_teams)]
fig = px.line(
    trend_plot,
    x="year",
    y="season_points",
    color="team_name",
    markers=True,
    title="Season Points Trend for Leading Teams",
)
fig.write_html(OUTPUT_CHARTS / "team_season_points_trend.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""volatility = (
    team_season.groupby("team_name", as_index=False)
    .agg(points_std=("season_points", "std"), points_mean=("season_points", "mean"), seasons=("year", "nunique"))
)
volatility["points_cv_pct"] = volatility["points_std"] / volatility["points_mean"] * 100
volatility = volatility.sort_values("points_cv_pct")
volatility.to_csv(OUTPUT_TABLES / "team_points_volatility.csv", index=False)
display(volatility)"""),
        md("""## 4. Home-Country Signal

Home advantage is a weak and sparse signal in this dataset because many constructors are multinational and not every home country appears every season. The analysis is recorded as a review signal, not a hard feature rule."""),
        code("""home_countries = {
    "Red Bull Racing": ["Austria"],
    "Ferrari": ["Italy"],
    "Mercedes": ["Great Britain", "Germany"],
    "McLaren": ["Great Britain"],
    "Aston Martin": ["Great Britain"],
    "Williams": ["Great Britain"],
    "Alpine": ["France"],
    "Haas F1 Team": ["United States"],
    "RB": ["Italy"],
    "Kick Sauber": ["Switzerland"],
    "Sauber": ["Switzerland"],
}
team_results["home_country_match"] = team_results.apply(
    lambda row: row["country_name"] in home_countries.get(row["team_name"], []),
    axis=1,
)
home_away = (
    team_results.dropna(subset=["finish_pos"])
    .groupby(["team_name", "home_country_match"], as_index=False)
    .agg(avg_finish=("finish_pos", "mean"), avg_points=("points", "mean"), rows=("driver_number", "count"))
)
home_pivot = home_away.pivot(index="team_name", columns="home_country_match", values=["avg_finish", "avg_points", "rows"])
home_pivot.columns = [f"{metric}_{'home' if flag else 'away'}" for metric, flag in home_pivot.columns]
home_pivot = home_pivot.reset_index()
if {"avg_finish_home", "avg_finish_away"}.issubset(home_pivot.columns):
    home_pivot["home_finish_delta"] = home_pivot["avg_finish_away"] - home_pivot["avg_finish_home"]
else:
    home_pivot["home_finish_delta"] = np.nan
home_pivot.to_csv(OUTPUT_TABLES / "home_country_signal.csv", index=False)
display(home_pivot.sort_values("home_finish_delta", ascending=False, na_position="last"))"""),
        code("""plot_home = home_pivot.dropna(subset=["home_finish_delta"]).copy()
fig = px.bar(
    plot_home.sort_values("home_finish_delta"),
    x="home_finish_delta",
    y="team_name",
    orientation="h",
    title="Home-Country Finish Delta (positive means better at home)",
)
fig.write_html(OUTPUT_CHARTS / "home_country_signal.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 5. Intra-Team Balance

A balanced team has two drivers contributing similar points. This section measures how much the lower-scoring driver contributes relative to the higher-scoring driver within each team."""),
        code("""driver_team_points = (
    team_results.groupby(["team_name", "driver_id"], as_index=False)
    .agg(driver_points=("points", "sum"), sessions=("session_key", "nunique"), avg_finish=("finish_pos", "mean"))
)
balance_rows = []
for team_name, group in driver_team_points.groupby("team_name"):
    active = group.sort_values("driver_points", ascending=False)
    if len(active) < 2:
        continue
    top_two = active.head(2)
    high = float(top_two.iloc[0]["driver_points"])
    low = float(top_two.iloc[1]["driver_points"])
    balance_rows.append({
        "team_name": team_name,
        "lead_driver": top_two.iloc[0]["driver_id"],
        "second_driver": top_two.iloc[1]["driver_id"],
        "lead_points": high,
        "second_points": low,
        "points_balance_pct": low / high * 100 if high else np.nan,
        "total_top_two_points": high + low,
    })
team_balance = pd.DataFrame(balance_rows).sort_values("points_balance_pct", ascending=False)
team_balance.to_csv(OUTPUT_TABLES / "team_driver_balance.csv", index=False)
display(team_balance)"""),
        code("""fig = px.bar(
    team_balance.sort_values("points_balance_pct"),
    x="points_balance_pct",
    y="team_name",
    orientation="h",
    color="total_top_two_points",
    title="Intra-Team Points Balance",
    labels={"points_balance_pct": "Second driver points as % of lead driver", "team_name": "Team"},
)
fig.write_html(OUTPUT_CHARTS / "team_driver_balance.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Final Team Analysis Report

The final report summarizes team structure signals for later Gold feature engineering."""),
        code("""report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "teams_analyzed": int(team_total["team_name"].nunique()),
    "sessions_analyzed": int(team_session["session_key"].nunique()),
    "total_points": total_points_all,
    "top1_points_concentration_pct": top1_points_pct,
    "top3_points_concentration_pct": top3_points_pct,
    "top_team": team_total.iloc[0]["team_name"] if len(team_total) else None,
    "top_team_points": float(team_total.iloc[0]["total_points"]) if len(team_total) else 0.0,
    "team_season_rows": int(len(team_season)),
    "home_signal_teams": int(home_pivot["home_finish_delta"].notna().sum()) if "home_finish_delta" in home_pivot else 0,
    "avg_driver_balance_pct": float(team_balance["points_balance_pct"].mean()) if not team_balance.empty else 0.0,
}
write_report("team_analysis", report)
write_insight(
    "Silver Team Analysis Insights",
    [
        f"Analyzed {report['teams_analyzed']} teams across {report['sessions_analyzed']} race/sprint sessions.",
        f"Top team: {report['top_team']} with {report['top_team_points']:.0f} points.",
        f"Top three teams account for {report['top3_points_concentration_pct']:.1f}% of points.",
        f"Average top-two driver balance is {report['avg_driver_balance_pct']:.1f}%.",
    ],
    [],
    [
        "Use team points tier, season trend, and intra-team balance as candidate Gold features.",
        "Treat home-country advantage as sparse review signal rather than a mandatory model feature.",
        "Keep team identity joined by session_key and driver_number to avoid driver/team mapping drift.",
    ],
)
(CHECKPOINTS / "silver_team_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "03_team_analysis.ipynb")

    circuit_setup = SETUP.replace('NOTEBOOK_NAME = "01_verify_cleaned_data"', 'NOTEBOOK_NAME = "04_circuit_analysis"')
    write_notebook([
        md("""# 04 Circuit Analysis

This notebook analyzes circuit-level characteristics: difficulty, lap-time evolution, overtaking frequency, weather context, and approximate circuit speed profile.

Telemetry is sampled from Parquet row groups only where needed; the notebook does not load full telemetry into memory."""),
        code(circuit_setup),
        code("""import pyarrow.parquet as pq

session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
sessions = pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet")
meetings = pd.read_parquet(CLEANED_DATA_PATH / "meetings.parquet")
laps = pd.read_parquet(CLEANED_DATA_PATH / "laps.parquet")
overtakes = pd.read_parquet(CLEANED_DATA_PATH / "overtakes.parquet")
weather = pd.read_parquet(CLEANED_DATA_PATH / "weather.parquet")

sessions["event_type"] = np.where(
    sessions["session_name"].astype(str).str.lower().eq("sprint"),
    "SPRINT_RACE",
    "GRAND_PRIX_RACE",
)
race_info = sessions[["session_key", "meeting_key", "year", "event_type", "circuit_short_name", "country_name"]].drop_duplicates()
print(f"Sessions: {race_info['session_key'].nunique()}")
print(f"Circuits: {race_info['circuit_short_name'].nunique()}")
print(f"Countries: {race_info['country_name'].nunique()}")"""),
        md("""## 1. Circuit Difficulty

Circuit difficulty is approximated with DNF rate and finish-position completeness. This is not a pure track-only measure, but it flags venues where race outcomes are more failure-prone in the observed data."""),
        code("""result_circuit = session_result.merge(race_info, on=["session_key", "meeting_key"], how="left")
result_circuit["finish_pos"] = pd.to_numeric(result_circuit["position"], errors="coerce")
result_circuit["is_dnf_like"] = result_circuit[["dnf", "dns", "dsq"]].any(axis=1) | result_circuit["finish_pos"].isna()
circuit_difficulty = (
    result_circuit.groupby("circuit_short_name", as_index=False)
    .agg(
        entries=("driver_number", "count"),
        sessions=("session_key", "nunique"),
        dnf_rate=("is_dnf_like", "mean"),
        avg_finish=("finish_pos", "mean"),
        countries=("country_name", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
    )
)
circuit_difficulty = circuit_difficulty[circuit_difficulty["entries"] >= 20].sort_values("dnf_rate", ascending=False)
circuit_difficulty.to_csv(OUTPUT_TABLES / "circuit_difficulty.csv", index=False)
display(circuit_difficulty.head(15))"""),
        code("""fig = px.bar(
    circuit_difficulty.head(15).sort_values("dnf_rate"),
    x="dnf_rate",
    y="circuit_short_name",
    orientation="h",
    title="Circuit Difficulty Proxy: DNF-like Rate",
    labels={"dnf_rate": "DNF-like rate", "circuit_short_name": "Circuit"},
)
fig.write_html(OUTPUT_CHARTS / "circuit_difficulty.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 2. Lap-Time Evolution by Circuit

For recurring circuits, fastest observed lap time by year is used as a lightweight proxy for pace evolution. This is sensitive to weather, race type, and regulation changes, so it should be interpreted as EDA context."""),
        code("""lap_circuit = laps.merge(race_info[["session_key", "year", "circuit_short_name", "event_type"]], on="session_key", how="left")
lap_circuit["lap_duration"] = pd.to_numeric(lap_circuit["lap_duration"], errors="coerce")
valid_laps = lap_circuit[lap_circuit["lap_duration"].between(50, 900)].copy()
circuit_year = (
    valid_laps.groupby(["circuit_short_name", "year"], as_index=False)
    .agg(
        fastest_lap=("lap_duration", "min"),
        median_lap=("lap_duration", "median"),
        avg_lap=("lap_duration", "mean"),
        laps=("lap_duration", "count"),
    )
)
circuit_year = circuit_year.sort_values(["circuit_short_name", "year"])
circuit_year["prev_fastest_lap"] = circuit_year.groupby("circuit_short_name")["fastest_lap"].shift(1)
circuit_year["fastest_lap_improvement_pct"] = (circuit_year["prev_fastest_lap"] - circuit_year["fastest_lap"]) / circuit_year["prev_fastest_lap"] * 100
circuit_year.to_csv(OUTPUT_TABLES / "circuit_lap_time_trends.csv", index=False)
display(circuit_year.head(30))"""),
        code("""repeat_circuits = circuit_year.groupby("circuit_short_name")["year"].nunique()
selected = repeat_circuits[repeat_circuits >= 2].index[:10]
trend = circuit_year[circuit_year["circuit_short_name"].isin(selected)]
fig = px.line(
    trend,
    x="year",
    y="fastest_lap",
    color="circuit_short_name",
    markers=True,
    title="Fastest Lap Trend by Recurring Circuit",
)
fig.write_html(OUTPUT_CHARTS / "circuit_lap_time_trends.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 3. Overtaking Frequency

Overtaking volume is aggregated per circuit and normalized by session count. This helps identify tracks that generate more on-track position changes."""),
        code("""overtakes_circuit = overtakes.merge(race_info[["session_key", "year", "circuit_short_name", "event_type"]], on="session_key", how="left")
session_overtakes = overtakes_circuit.groupby(["session_key", "circuit_short_name"], as_index=False).size().rename(columns={"size": "overtakes"})
circuit_overtakes = (
    session_overtakes.groupby("circuit_short_name", as_index=False)
    .agg(avg_overtakes=("overtakes", "mean"), median_overtakes=("overtakes", "median"), sessions=("session_key", "nunique"))
    .sort_values("avg_overtakes", ascending=False)
)
circuit_overtakes.to_csv(OUTPUT_TABLES / "circuit_overtakes.csv", index=False)
display(circuit_overtakes.head(15))"""),
        code("""fig = px.bar(
    circuit_overtakes.head(15).sort_values("avg_overtakes"),
    x="avg_overtakes",
    y="circuit_short_name",
    orientation="h",
    title="Average Overtakes by Circuit",
)
fig.write_html(OUTPUT_CHARTS / "circuit_overtakes.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 4. Circuit Weather Context

Weather is summarized by circuit to expose environmental differences that can later interact with tyre and pace features."""),
        code("""weather_circuit = weather.merge(race_info[["session_key", "circuit_short_name", "country_name"]], on="session_key", how="left")
weather_cols = [column for column in ["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall"] if column in weather_circuit.columns]
circuit_weather = weather_circuit.groupby("circuit_short_name", as_index=False)[weather_cols].mean(numeric_only=True)
circuit_weather.to_csv(OUTPUT_TABLES / "circuit_weather.csv", index=False)
display(circuit_weather.sort_values("track_temperature", ascending=False).head(15))"""),
        code("""fig = px.scatter(
    circuit_weather,
    x="air_temperature",
    y="track_temperature",
    size="rainfall" if "rainfall" in circuit_weather.columns else None,
    hover_name="circuit_short_name",
    title="Circuit Weather Profile",
)
fig.write_html(OUTPUT_CHARTS / "circuit_weather_profile.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 5. Circuit Speed Profile from Telemetry Sample

To avoid memory pressure, speed profile uses a bounded sample from `car_data.parquet` row groups. This is enough for EDA-level circuit classification, not a final feature store."""),
        code("""def sample_car_data(max_row_groups: int = 8, rows_per_group: int = 30_000) -> pd.DataFrame:
    path = CLEANED_DATA_PATH / "car_data.parquet"
    parquet_file = pq.ParquetFile(path)
    frames = []
    columns = [column for column in ["session_key", "driver_number", "Speed", "Throttle", "Brake", "DRS", "lap_number"] if column in parquet_file.schema.names]
    step = max(parquet_file.num_row_groups // max_row_groups, 1)
    for row_group_idx in range(0, parquet_file.num_row_groups, step):
        if len(frames) >= max_row_groups:
            break
        chunk = parquet_file.read_row_group(row_group_idx, columns=columns).to_pandas()
        if len(chunk) > rows_per_group:
            chunk = chunk.sample(rows_per_group, random_state=42)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

car_sample = sample_car_data()
speed_col = "Speed" if "Speed" in car_sample.columns else "speed"
car_sample[speed_col] = pd.to_numeric(car_sample[speed_col], errors="coerce")
car_sample = car_sample.merge(race_info[["session_key", "circuit_short_name"]], on="session_key", how="left")
circuit_speed = (
    car_sample.dropna(subset=["circuit_short_name", speed_col])
    .groupby("circuit_short_name", as_index=False)
    .agg(avg_speed_kmh=(speed_col, "mean"), p95_speed_kmh=(speed_col, lambda s: s.quantile(0.95)), samples=(speed_col, "count"))
    .sort_values("avg_speed_kmh", ascending=False)
)
if len(circuit_speed) >= 3:
    ranks = circuit_speed["avg_speed_kmh"].rank(method="first")
    circuit_speed["speed_profile"] = pd.qcut(ranks, q=3, labels=["Low-speed", "Mixed", "High-speed"])
else:
    circuit_speed["speed_profile"] = "Sampled"
circuit_speed.to_csv(OUTPUT_TABLES / "circuit_speed_profile.csv", index=False)
display(circuit_speed)"""),
        code("""fig = px.bar(
    circuit_speed.sort_values("avg_speed_kmh"),
    x="avg_speed_kmh",
    y="circuit_short_name",
    color="speed_profile",
    orientation="h",
    title="Sampled Telemetry Speed Profile by Circuit",
)
fig.write_html(OUTPUT_CHARTS / "circuit_speed_profile.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Final Circuit Analysis Report"""),
        code("""report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "circuits_analyzed": int(race_info["circuit_short_name"].nunique()),
    "sessions_analyzed": int(race_info["session_key"].nunique()),
    "highest_dnf_circuit": circuit_difficulty.iloc[0]["circuit_short_name"] if len(circuit_difficulty) else None,
    "highest_dnf_rate": float(circuit_difficulty.iloc[0]["dnf_rate"]) if len(circuit_difficulty) else 0.0,
    "highest_overtake_circuit": circuit_overtakes.iloc[0]["circuit_short_name"] if len(circuit_overtakes) else None,
    "highest_avg_overtakes": float(circuit_overtakes.iloc[0]["avg_overtakes"]) if len(circuit_overtakes) else 0.0,
    "speed_profile_circuits": int(len(circuit_speed)),
}
write_report("circuit_analysis", report)
write_insight(
    "Silver Circuit Analysis Insights",
    [
        f"Analyzed {report['circuits_analyzed']} circuits across {report['sessions_analyzed']} sessions.",
        f"Highest DNF-like circuit: {report['highest_dnf_circuit']} ({report['highest_dnf_rate']:.1%}).",
        f"Highest overtake circuit: {report['highest_overtake_circuit']} ({report['highest_avg_overtakes']:.1f} avg overtakes).",
    ],
    [],
    [
        "Use circuit DNF rate, overtake frequency, weather profile, and sampled speed profile as candidate Gold context features.",
        "Keep telemetry-derived circuit speed profile as approximate until a full feature pipeline aggregates all row groups.",
    ],
)
(CHECKPOINTS / "silver_circuit_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "04_circuit_analysis.ipynb")

    weather_setup = SETUP.replace('NOTEBOOK_NAME = "01_verify_cleaned_data"', 'NOTEBOOK_NAME = "05_weather_analysis"')
    write_notebook([
        md("""# 05 Weather Analysis

This notebook studies race weather patterns and their relationship with lap time and race outcomes.

The objective is to decide which environmental features should be retained for Gold feature engineering."""),
        code(weather_setup),
        code("""weather = pd.read_parquet(CLEANED_DATA_PATH / "weather.parquet")
sessions = pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet")
laps = pd.read_parquet(CLEANED_DATA_PATH / "laps.parquet")
session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")

sessions["event_type"] = np.where(
    sessions["session_name"].astype(str).str.lower().eq("sprint"),
    "SPRINT_RACE",
    "GRAND_PRIX_RACE",
)
session_context = sessions[["session_key", "year", "event_type", "circuit_short_name", "country_name", "date_start"]].drop_duplicates()
weather_race = weather.merge(session_context, on="session_key", how="left")
weather_race["date"] = pd.to_datetime(weather_race["date"], errors="coerce", utc=True) if "date" in weather_race.columns else pd.NaT
print(f"Weather rows: {len(weather_race):,}")
print(f"Sessions with weather: {weather_race['session_key'].nunique()}")
print(f"Circuits: {weather_race['circuit_short_name'].nunique()}")"""),
        md("""## 1. Weather Variable Distributions

Weather values are continuous race-context signals. The key question is whether they have enough variation to matter for tyre and pace modeling."""),
        code("""weather_cols = [column for column in ["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall"] if column in weather_race.columns]
weather_summary = weather_race[weather_cols].describe().round(3).reset_index().rename(columns={"index": "metric"})
weather_summary.to_csv(OUTPUT_TABLES / "weather_distribution_summary.csv", index=False)
display(weather_summary)"""),
        code("""plot_cols = [column for column in ["air_temperature", "track_temperature", "humidity", "wind_speed"] if column in weather_race.columns]
plot_df = weather_race[plot_cols].melt(var_name="weather_variable", value_name="value").dropna()
fig = px.box(
    plot_df,
    x="weather_variable",
    y="value",
    color="weather_variable",
    title="Weather Variable Distributions",
)
fig.write_html(OUTPUT_CHARTS / "weather_distributions.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 2. Wet vs Dry Sessions

Rainfall is sparse but strategically important. This section summarizes how often wet conditions appear at session level."""),
        code("""session_weather = weather_race.groupby("session_key", as_index=False).agg(
    air_temperature=("air_temperature", "mean"),
    track_temperature=("track_temperature", "mean"),
    humidity=("humidity", "mean"),
    pressure=("pressure", "mean"),
    wind_speed=("wind_speed", "mean"),
    rainfall=("rainfall", "mean"),
    year=("year", "first"),
    event_type=("event_type", "first"),
    circuit_short_name=("circuit_short_name", "first"),
    country_name=("country_name", "first"),
    date_start=("date_start", "first"),
)
session_weather["is_wet_session"] = session_weather["rainfall"].fillna(0) > 0
wet_summary = session_weather.groupby(["year", "event_type"], as_index=False).agg(
    sessions=("session_key", "count"),
    wet_sessions=("is_wet_session", "sum"),
    wet_rate=("is_wet_session", "mean"),
)
wet_summary.to_csv(OUTPUT_TABLES / "wet_session_summary.csv", index=False)
display(wet_summary)"""),
        code("""fig = px.bar(
    wet_summary,
    x="year",
    y="wet_rate",
    color="event_type",
    barmode="group",
    title="Wet Session Rate by Year and Event Type",
)
fig.write_html(OUTPUT_CHARTS / "wet_session_rate.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 3. Weather vs Lap Time

This joins session-level weather to session-level lap metrics to measure whether environmental conditions correlate with pace."""),
        code("""lap_session = laps.copy()
lap_session["lap_duration"] = pd.to_numeric(lap_session["lap_duration"], errors="coerce")
lap_session = lap_session[lap_session["lap_duration"].between(50, 900)]
lap_session_summary = lap_session.groupby("session_key", as_index=False).agg(
    median_lap=("lap_duration", "median"),
    fastest_lap=("lap_duration", "min"),
    lap_std=("lap_duration", "std"),
    laps=("lap_duration", "count"),
)
weather_lap = session_weather.merge(lap_session_summary, on="session_key", how="inner")
weather_lap.to_csv(OUTPUT_TABLES / "weather_lap_session_features.csv", index=False)
weather_lap_corr = weather_lap[[column for column in ["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall", "median_lap", "fastest_lap", "lap_std"] if column in weather_lap.columns]].corr()
weather_lap_corr.to_csv(OUTPUT_TABLES / "weather_lap_correlations.csv")
display(weather_lap_corr.round(3))"""),
        code("""fig = px.scatter(
    weather_lap,
    x="track_temperature",
    y="median_lap",
    color="event_type",
    hover_name="circuit_short_name",
    title="Track Temperature vs Median Lap Time",
)
if len(weather_lap.dropna(subset=["track_temperature", "median_lap"])) > 2:
    fit_data = weather_lap.dropna(subset=["track_temperature", "median_lap"])
    fit = np.polyfit(fit_data["track_temperature"], fit_data["median_lap"], deg=1)
    x_line = np.array([fit_data["track_temperature"].min(), fit_data["track_temperature"].max()])
    fig.add_scatter(x=x_line, y=fit[0] * x_line + fit[1], mode="lines", name="Linear fit")
fig.write_html(OUTPUT_CHARTS / "track_temp_vs_lap_time.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""fig = px.imshow(
    weather_lap_corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Weather and Lap-Time Correlation Matrix",
)
fig.write_html(OUTPUT_CHARTS / "weather_lap_correlation_heatmap.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 4. Circuit Weather Profiles

Aggregating weather by circuit identifies venues with systematically hotter, wetter, or more humid conditions."""),
        code("""circuit_weather = session_weather.groupby("circuit_short_name", as_index=False).agg(
    avg_air_temperature=("air_temperature", "mean"),
    avg_track_temperature=("track_temperature", "mean"),
    avg_humidity=("humidity", "mean"),
    avg_pressure=("pressure", "mean"),
    avg_wind_speed=("wind_speed", "mean"),
    wet_rate=("is_wet_session", "mean"),
    sessions=("session_key", "count"),
).sort_values("avg_track_temperature", ascending=False)
circuit_weather.to_csv(OUTPUT_TABLES / "circuit_weather_profiles.csv", index=False)
display(circuit_weather.head(15))"""),
        code("""fig = px.scatter(
    circuit_weather,
    x="avg_air_temperature",
    y="avg_track_temperature",
    size="wet_rate",
    color="avg_humidity",
    hover_name="circuit_short_name",
    title="Circuit Weather Profiles",
)
fig.write_html(OUTPUT_CHARTS / "circuit_weather_profiles.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 5. Seasonal Weather Patterns

Month-level aggregation helps reveal whether calendar placement introduces systematic weather differences."""),
        code("""session_weather["month"] = pd.to_datetime(session_weather["date_start"], errors="coerce", utc=True).dt.month
monthly_weather = session_weather.groupby("month", as_index=False).agg(
    air_temperature=("air_temperature", "mean"),
    track_temperature=("track_temperature", "mean"),
    humidity=("humidity", "mean"),
    rainfall=("rainfall", "mean"),
    sessions=("session_key", "count"),
)
monthly_weather.to_csv(OUTPUT_TABLES / "monthly_weather.csv", index=False)
display(monthly_weather)"""),
        code("""fig = px.line(
    monthly_weather,
    x="month",
    y=["air_temperature", "track_temperature"],
    markers=True,
    title="Seasonal Temperature Pattern",
)
fig.write_html(OUTPUT_CHARTS / "seasonal_temperature.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 6. Weather vs Race Outcomes

Weather is joined to race results to inspect whether conditions correlate with finishing position, DNF-like outcomes, or points."""),
        code("""outcome = session_result.merge(session_weather, on="session_key", how="left")
outcome["finish_pos"] = pd.to_numeric(outcome["position"], errors="coerce")
outcome["is_dnf_like"] = outcome[["dnf", "dns", "dsq"]].any(axis=1) | outcome["finish_pos"].isna()
outcome_cols = [column for column in ["finish_pos", "points", "is_dnf_like"] if column in outcome.columns]
weather_vars = [column for column in ["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall"] if column in outcome.columns]
rows = []
for weather_var in weather_vars:
    for outcome_col in outcome_cols:
        series = outcome[outcome_col].astype(float) if outcome_col == "is_dnf_like" else pd.to_numeric(outcome[outcome_col], errors="coerce")
        rows.append({"weather_var": weather_var, "outcome": outcome_col, "correlation": outcome[weather_var].corr(series)})
weather_outcome_corr = pd.DataFrame(rows)
weather_outcome_corr.to_csv(OUTPUT_TABLES / "weather_outcome_correlations.csv", index=False)
display(weather_outcome_corr.pivot(index="weather_var", columns="outcome", values="correlation").round(3))"""),
        code("""corr_pivot = weather_outcome_corr.pivot(index="weather_var", columns="outcome", values="correlation")
fig = px.imshow(
    corr_pivot,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Weather vs Race Outcome Correlations",
)
fig.write_html(OUTPUT_CHARTS / "weather_outcome_correlation_heatmap.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Final Weather Analysis Report"""),
        code("""temp_lap_corr = float(weather_lap["track_temperature"].corr(weather_lap["median_lap"])) if {"track_temperature", "median_lap"}.issubset(weather_lap.columns) else float("nan")
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "weather_rows": int(len(weather_race)),
    "sessions_with_weather": int(session_weather["session_key"].nunique()),
    "wet_sessions": int(session_weather["is_wet_session"].sum()),
    "wet_session_rate_pct": float(session_weather["is_wet_session"].mean() * 100),
    "avg_air_temperature": float(session_weather["air_temperature"].mean()),
    "avg_track_temperature": float(session_weather["track_temperature"].mean()),
    "track_temp_median_lap_corr": temp_lap_corr,
}
write_report("weather_analysis", report)
write_insight(
    "Silver Weather Analysis Insights",
    [
        f"Analyzed {report['sessions_with_weather']} sessions with weather data.",
        f"Wet session rate: {report['wet_session_rate_pct']:.1f}%.",
        f"Track temperature vs median lap correlation: {report['track_temp_median_lap_corr']:.3f}.",
    ],
    [],
    [
        "Keep track_temperature, air_temperature, humidity, pressure, wind_speed, and rainfall as Gold race-context candidates.",
        "Treat rainfall as sparse but strategically important rather than dropping it for low frequency.",
        "Use circuit-level weather summaries to enrich race context features.",
    ],
)
(CHECKPOINTS / "silver_weather_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "05_weather_analysis.ipynb")

    telemetry_setup = SETUP.replace('NOTEBOOK_NAME = "01_verify_cleaned_data"', 'NOTEBOOK_NAME = "06_telemetry_analysis"')
    write_notebook([
        md("""# 06 Telemetry Analysis

This notebook analyzes sampled telemetry patterns from `car_data.parquet` and `location.parquet`.

The Silver telemetry files are large, so this notebook samples Parquet row groups and aggregates immediately. The output is for EDA and feature design, not a full telemetry feature store."""),
        code(telemetry_setup),
        code("""import pyarrow.parquet as pq

session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
sessions = pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet")
drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
sessions["event_type"] = np.where(
    sessions["session_name"].astype(str).str.lower().eq("sprint"),
    "SPRINT_RACE",
    "GRAND_PRIX_RACE",
)
driver_dim = drivers[["session_key", "driver_number", "full_name", "team_name"]].drop_duplicates(["session_key", "driver_number"])

def sample_parquet(path: Path, columns: list[str], max_row_groups: int = 10, rows_per_group: int = 40_000) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    available = parquet_file.schema.names
    selected = [column for column in columns if column in available]
    frames = []
    step = max(parquet_file.num_row_groups // max_row_groups, 1)
    for row_group_idx in range(0, parquet_file.num_row_groups, step):
        if len(frames) >= max_row_groups:
            break
        chunk = parquet_file.read_row_group(row_group_idx, columns=selected).to_pandas()
        if len(chunk) > rows_per_group:
            chunk = chunk.sample(rows_per_group, random_state=42)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

car_sample = sample_parquet(
    CLEANED_DATA_PATH / "car_data.parquet",
    ["session_key", "driver_number", "Speed", "RPM", "nGear", "Throttle", "Brake", "DRS", "lap_number", "compound"],
)
loc_sample = sample_parquet(
    CLEANED_DATA_PATH / "location.parquet",
    ["session_key", "driver_number", "X", "Y", "Z", "lap_number"],
    max_row_groups=6,
    rows_per_group=30_000,
)
print(f"Car telemetry sample rows: {len(car_sample):,}")
print(f"Location telemetry sample rows: {len(loc_sample):,}")
print(f"Car sessions sampled: {car_sample['session_key'].nunique() if not car_sample.empty else 0}")"""),
        md("""## 1. Telemetry Metric Distributions

This section profiles the core car channels: speed, RPM, gear, throttle, brake, and DRS. These distributions guide scaling and clipping decisions for Gold features."""),
        code("""numeric_cols = [column for column in ["Speed", "RPM", "nGear", "Throttle", "DRS", "lap_number"] if column in car_sample.columns]
for column in numeric_cols:
    car_sample[column] = pd.to_numeric(car_sample[column], errors="coerce")
if "Brake" in car_sample.columns:
    car_sample["Brake"] = car_sample["Brake"].astype(float)
telemetry_summary = car_sample[[column for column in numeric_cols + ["Brake"] if column in car_sample.columns]].describe().round(3).reset_index().rename(columns={"index": "metric"})
telemetry_summary.to_csv(OUTPUT_TABLES / "telemetry_distribution_summary.csv", index=False)
display(telemetry_summary)"""),
        code("""plot_cols = [column for column in ["Speed", "RPM", "Throttle", "Brake", "DRS"] if column in car_sample.columns]
plot_df = car_sample[plot_cols].melt(var_name="channel", value_name="value").dropna()
fig = px.box(plot_df, x="channel", y="value", color="channel", title="Sampled Telemetry Channel Distributions")
fig.write_html(OUTPUT_CHARTS / "telemetry_distributions.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 2. Driver-Level Telemetry Profiles

Telemetry is aggregated by driver across sampled sessions. These features are prototypes for Gold `driver_telemetry_features`."""),
        code("""driver_telemetry = (
    car_sample.groupby(["session_key", "driver_number"], as_index=False)
    .agg(
        avg_speed=("Speed", "mean"),
        max_speed=("Speed", "max"),
        std_speed=("Speed", "std"),
        avg_rpm=("RPM", "mean"),
        avg_throttle=("Throttle", "mean"),
        avg_brake=("Brake", "mean"),
        drs_rate=("DRS", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()),
        sampled_rows=("Speed", "count"),
    )
)
driver_telemetry = driver_telemetry.merge(driver_dim, on=["session_key", "driver_number"], how="left")
driver_telemetry["driver_id"] = driver_telemetry["full_name"].fillna("Driver " + driver_telemetry["driver_number"].astype(str))
driver_profile = driver_telemetry.groupby("driver_id", as_index=False).agg(
    avg_speed=("avg_speed", "mean"),
    max_speed=("max_speed", "max"),
    avg_throttle=("avg_throttle", "mean"),
    avg_brake=("avg_brake", "mean"),
    drs_rate=("drs_rate", "mean"),
    sampled_sessions=("session_key", "nunique"),
    sampled_rows=("sampled_rows", "sum"),
).sort_values("avg_speed", ascending=False)
driver_profile.to_csv(OUTPUT_TABLES / "driver_telemetry_profile.csv", index=False)
display(driver_profile.head(20))"""),
        code("""fig = px.scatter(
    driver_profile,
    x="avg_throttle",
    y="avg_speed",
    size="sampled_rows",
    color="drs_rate",
    hover_name="driver_id",
    title="Driver Telemetry Profile: Throttle vs Speed",
)
fig.write_html(OUTPUT_CHARTS / "driver_telemetry_profile.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 3. Lap-Type Prototype

Lap-type labels are approximated from sampled telemetry. This is intentionally simple and should be refined in Gold with full-lap context."""),
        code("""lap_telemetry = (
    car_sample.dropna(subset=["lap_number"])
    .groupby(["session_key", "driver_number", "lap_number"], as_index=False)
    .agg(
        avg_speed=("Speed", "mean"),
        max_speed=("Speed", "max"),
        avg_throttle=("Throttle", "mean"),
        avg_brake=("Brake", "mean"),
        rows=("Speed", "count"),
    )
)
lap_telemetry["lap_type"] = "normal"
lap_telemetry.loc[lap_telemetry["lap_number"].eq(1), "lap_type"] = "out_lap"
lap_telemetry.loc[lap_telemetry["avg_speed"].lt(100), "lap_type"] = "slow_or_pit_lap"
fastest_speed = lap_telemetry.groupby(["session_key", "driver_number"])["avg_speed"].transform("max")
lap_telemetry.loc[lap_telemetry["avg_speed"].eq(fastest_speed), "lap_type"] = "sample_fast_lap"
lap_type_summary = lap_telemetry.groupby("lap_type", as_index=False).agg(
    laps=("lap_number", "count"),
    avg_speed=("avg_speed", "mean"),
    avg_throttle=("avg_throttle", "mean"),
    avg_brake=("avg_brake", "mean"),
)
lap_telemetry.to_csv(OUTPUT_TABLES / "sample_lap_telemetry.csv", index=False)
lap_type_summary.to_csv(OUTPUT_TABLES / "lap_type_summary.csv", index=False)
display(lap_type_summary)"""),
        code("""fig = px.bar(
    lap_type_summary,
    x="lap_type",
    y="laps",
    color="avg_speed",
    title="Sampled Lap-Type Distribution",
)
fig.write_html(OUTPUT_CHARTS / "lap_type_summary.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 4. Location Sample Coverage

Location telemetry is used here only for coordinate spread and data sanity. Full racing-line features should be built in Gold with more deliberate spatial aggregation."""),
        code("""for column in ["X", "Y", "Z", "lap_number"]:
    if column in loc_sample.columns:
        loc_sample[column] = pd.to_numeric(loc_sample[column], errors="coerce")
location_summary = loc_sample[[column for column in ["X", "Y", "Z", "lap_number"] if column in loc_sample.columns]].describe().round(3).reset_index().rename(columns={"index": "metric"})
location_summary.to_csv(OUTPUT_TABLES / "location_distribution_summary.csv", index=False)
display(location_summary)"""),
        code("""loc_plot = loc_sample.dropna(subset=["X", "Y"]).sample(min(20_000, len(loc_sample)), random_state=42) if {"X", "Y"}.issubset(loc_sample.columns) and len(loc_sample) else pd.DataFrame()
fig = px.scatter(
    loc_plot,
    x="X",
    y="Y",
    color="session_key" if "session_key" in loc_plot.columns else None,
    title="Sampled GPS Position Coverage",
    render_mode="webgl",
)
fig.write_html(OUTPUT_CHARTS / "location_sample_coverage.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 5. Telemetry vs Race Outcome

Sampled telemetry is joined to finishing results to inspect whether aggregate car channels show predictive signal."""),
        code("""telemetry_results = driver_telemetry.merge(
    session_result[["session_key", "driver_number", "position", "points"]],
    on=["session_key", "driver_number"],
    how="inner",
)
telemetry_results["finish_pos"] = pd.to_numeric(telemetry_results["position"], errors="coerce")
feature_cols = [column for column in ["avg_speed", "max_speed", "std_speed", "avg_rpm", "avg_throttle", "avg_brake", "drs_rate"] if column in telemetry_results.columns]
correlations = []
for column in feature_cols:
    correlations.append({"feature": column, "correlation_with_finish": telemetry_results[column].corr(telemetry_results["finish_pos"])})
telemetry_corr = pd.DataFrame(correlations).sort_values("correlation_with_finish")
telemetry_results.to_csv(OUTPUT_TABLES / "telemetry_result_join.csv", index=False)
telemetry_corr.to_csv(OUTPUT_TABLES / "telemetry_finish_correlations.csv", index=False)
display(telemetry_corr)"""),
        code("""fig = px.bar(
    telemetry_corr,
    x="correlation_with_finish",
    y="feature",
    orientation="h",
    title="Sampled Telemetry Correlation with Finish Position",
)
fig.write_html(OUTPUT_CHARTS / "telemetry_finish_correlations.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Final Telemetry Analysis Report"""),
        code("""top_corr = telemetry_corr.iloc[telemetry_corr["correlation_with_finish"].abs().argmax()] if len(telemetry_corr) else None
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "car_sample_rows": int(len(car_sample)),
    "location_sample_rows": int(len(loc_sample)),
    "sampled_driver_sessions": int(driver_telemetry[["session_key", "driver_number"]].drop_duplicates().shape[0]),
    "lap_telemetry_rows": int(len(lap_telemetry)),
    "top_telemetry_corr_feature": top_corr["feature"] if top_corr is not None else None,
    "top_telemetry_corr_value": float(top_corr["correlation_with_finish"]) if top_corr is not None else 0.0,
}
write_report("telemetry_analysis", report)
write_insight(
    "Silver Telemetry Analysis Insights",
    [
        f"Sampled {report['car_sample_rows']:,} car telemetry rows and {report['location_sample_rows']:,} location rows.",
        f"Aggregated {report['sampled_driver_sessions']} driver-session telemetry profiles.",
        f"Strongest sampled telemetry/finish correlation: {report['top_telemetry_corr_feature']} = {report['top_telemetry_corr_value']:.3f}.",
    ],
    [],
    [
        "Use full row-group aggregation in Gold for final telemetry features; this notebook only validates signal direction.",
        "Keep avg_speed, throttle, brake, DRS rate, and lap-type features as candidate telemetry features.",
        "Use location data carefully; sampled GPS coverage is useful for sanity checks but not final racing-line modeling.",
    ],
)
(CHECKPOINTS / "silver_telemetry_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "06_telemetry_analysis.ipynb")

    correlation_setup = SETUP.replace('NOTEBOOK_NAME = "01_verify_cleaned_data"', 'NOTEBOOK_NAME = "07_correlation_analysis"')
    write_notebook([
        md("""# 07 Correlation Analysis

This notebook closes the Silver EDA layer by building a compact driver-session feature matrix and checking the strongest relationships before Gold feature engineering.

The goal is not to train a production model here. The goal is to identify useful signals, redundant variables, and possible leakage risks while keeping telemetry usage memory-safe."""),
        code(correlation_setup),
        md("""## 1. Compact Driver-Session Feature Matrix

The matrix is built at one row per `session_key + driver_number`. This grain is the correct bridge between cleaned operational data and Gold modeling because race outcomes, grid position, lap summaries, stint behavior, weather context, and sampled telemetry can all be aligned without reading full telemetry files."""),
        code("""session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
sessions = pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet")
meetings = pd.read_parquet(CLEANED_DATA_PATH / "meetings.parquet")
laps = pd.read_parquet(CLEANED_DATA_PATH / "laps.parquet")
starting_grid = pd.read_parquet(CLEANED_DATA_PATH / "starting_grid.parquet")
weather = pd.read_parquet(CLEANED_DATA_PATH / "weather.parquet")
stints = pd.read_parquet(CLEANED_DATA_PATH / "stints.parquet")
overtakes = pd.read_parquet(CLEANED_DATA_PATH / "overtakes.parquet")

sessions["event_type"] = np.where(
    sessions["session_name"].astype(str).str.lower().eq("sprint"),
    "SPRINT_RACE",
    "GRAND_PRIX_RACE",
)

base = session_result.copy()
base["finish_pos"] = pd.to_numeric(base["position"], errors="coerce")
base["target_win"] = (base["finish_pos"] == 1).astype(int)
base["target_podium"] = (base["finish_pos"] <= 3).astype(int)
base["target_top10"] = (base["finish_pos"] <= 10).astype(int)
base["is_dnf_like"] = base[["dnf", "dns", "dsq"]].any(axis=1) | base["finish_pos"].isna()

driver_dim = drivers[["session_key", "driver_number", "full_name", "team_name"]].drop_duplicates(["session_key", "driver_number"])
base = base.merge(driver_dim, on=["session_key", "driver_number"], how="left")
base = base.merge(sessions[["session_key", "year", "event_type", "meeting_key", "circuit_short_name", "country_name"]], on="session_key", how="left", suffixes=("", "_session"))
base["driver_id"] = base["full_name"].fillna("Driver " + base["driver_number"].astype(str))

lap_work = laps.copy()
lap_work["lap_duration"] = pd.to_numeric(lap_work["lap_duration"], errors="coerce")
lap_work = lap_work[lap_work["lap_duration"].between(50, 900)]
lap_features = lap_work.groupby(["session_key", "driver_number"], as_index=False).agg(
    avg_lap=("lap_duration", "mean"),
    median_lap=("lap_duration", "median"),
    best_lap=("lap_duration", "min"),
    std_lap=("lap_duration", "std"),
    laps_completed=("lap_number", "nunique"),
    outlier_laps=("is_outlier_lap", "sum"),
    pit_out_laps=("is_pit_out_lap", "sum"),
    avg_i1_speed=("i1_speed", "mean"),
    avg_i2_speed=("i2_speed", "mean"),
    avg_st_speed=("st_speed", "mean"),
)

grid = starting_grid[["session_key", "driver_number", "position"]].copy()
grid["grid_pos"] = pd.to_numeric(grid["position"], errors="coerce")
grid = grid.drop(columns=["position"])

weather_session = weather.groupby("session_key", as_index=False).agg(
    air_temperature=("air_temperature", "mean"),
    track_temperature=("track_temperature", "mean"),
    humidity=("humidity", "mean"),
    pressure=("pressure", "mean"),
    wind_speed=("wind_speed", "mean"),
    rainfall=("rainfall", "max"),
)

stint_features = stints.groupby(["session_key", "driver_number"], as_index=False).agg(
    stint_count=("stint_number", "nunique"),
    max_tyre_age_start=("tyre_age_at_start", "max"),
    avg_tyre_age_start=("tyre_age_at_start", "mean"),
)

overtake_for = overtakes.groupby(["session_key", "overtaking_driver_number"]).size().reset_index(name="overtakes_made")
overtake_for = overtake_for.rename(columns={"overtaking_driver_number": "driver_number"})
overtake_against = overtakes.groupby(["session_key", "overtaken_driver_number"]).size().reset_index(name="overtaken_count")
overtake_against = overtake_against.rename(columns={"overtaken_driver_number": "driver_number"})

master = (
    base
    .merge(lap_features, on=["session_key", "driver_number"], how="left")
    .merge(grid, on=["session_key", "driver_number"], how="left")
    .merge(weather_session, on="session_key", how="left")
    .merge(stint_features, on=["session_key", "driver_number"], how="left")
    .merge(overtake_for, on=["session_key", "driver_number"], how="left")
    .merge(overtake_against, on=["session_key", "driver_number"], how="left")
)
master[["overtakes_made", "overtaken_count"]] = master[["overtakes_made", "overtaken_count"]].fillna(0)
master["positions_gained"] = master["grid_pos"] - master["finish_pos"]
master["points_per_lap"] = master["points"] / master["laps_completed"].replace(0, np.nan)

telemetry_path = ROOT / "eda" / "silver" / "outputs" / "tables" / "06_telemetry_analysis" / "telemetry_result_join.csv"
if telemetry_path.exists():
    telemetry_sample = pd.read_csv(telemetry_path)
    telemetry_cols = [
        "session_key", "driver_number", "avg_speed", "max_speed", "std_speed",
        "avg_rpm", "avg_throttle", "avg_brake", "drs_rate"
    ]
    telemetry_sample = telemetry_sample[[column for column in telemetry_cols if column in telemetry_sample.columns]]
    master = master.merge(telemetry_sample, on=["session_key", "driver_number"], how="left")

master.to_csv(OUTPUT_TABLES / "silver_driver_session_feature_matrix.csv", index=False)
print(f"Feature matrix rows: {len(master):,}")
print(f"Feature matrix columns: {master.shape[1]:,}")
display(master.head(12))"""),
        md("""## 2. Correlation Matrix

Correlation is used as a first-pass diagnostic. Negative correlation with `finish_pos` means the feature is associated with better finishing positions because P1 is numerically lower than P20. The target columns are excluded from feature-candidate correlation to avoid treating labels as model inputs."""),
        code("""candidate_features = [
    "grid_pos", "avg_lap", "median_lap", "best_lap", "std_lap", "laps_completed",
    "outlier_laps", "pit_out_laps", "avg_i1_speed", "avg_i2_speed", "avg_st_speed",
    "air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall",
    "stint_count", "max_tyre_age_start", "avg_tyre_age_start",
    "overtakes_made", "overtaken_count", "positions_gained",
    "avg_speed", "max_speed", "std_speed", "avg_rpm", "avg_throttle", "avg_brake", "drs_rate",
]
available_features = [column for column in candidate_features if column in master.columns]
numeric_cols = available_features + ["finish_pos", "points", "target_win", "target_podium", "target_top10", "is_dnf_like"]
numeric = master[numeric_cols].apply(pd.to_numeric, errors="coerce")
corr_matrix = numeric.corr()
corr_matrix.to_csv(OUTPUT_TABLES / "silver_feature_correlation_matrix.csv")

finish_corr = (
    corr_matrix["finish_pos"]
    .drop(labels=["finish_pos"], errors="ignore")
    .dropna()
    .sort_values()
    .reset_index()
)
finish_corr.columns = ["feature", "correlation_with_finish_pos"]
finish_corr.to_csv(OUTPUT_TABLES / "finish_position_correlations.csv", index=False)
display(finish_corr)"""),
        code("""heatmap_cols = [column for column in [
    "finish_pos", "grid_pos", "avg_lap", "best_lap", "std_lap", "laps_completed",
    "track_temperature", "rainfall", "stint_count", "overtakes_made", "overtaken_count",
    "avg_speed", "avg_throttle", "drs_rate"
] if column in corr_matrix.columns]
fig = px.imshow(
    corr_matrix.loc[heatmap_cols, heatmap_cols],
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Silver Feature Correlation Heatmap",
)
fig.write_html(OUTPUT_CHARTS / "feature_correlation_heatmap.html", include_plotlyjs="cdn")
fig.show()"""),
        code("""plot_corr = finish_corr.copy()
plot_corr["abs_corr"] = plot_corr["correlation_with_finish_pos"].abs()
plot_corr = plot_corr.sort_values("abs_corr", ascending=False).head(20).sort_values("correlation_with_finish_pos")
fig = px.bar(
    plot_corr,
    x="correlation_with_finish_pos",
    y="feature",
    orientation="h",
    color="correlation_with_finish_pos",
    color_continuous_scale="RdBu_r",
    title="Top Correlations with Finish Position",
)
fig.write_html(OUTPUT_CHARTS / "finish_position_correlations.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 3. Winner, Podium, and Top-10 Feature Contrast

This contrast turns correlations into interpretable race patterns. It compares robust medians for winners, podium finishers, top-10 finishers, and the rest of the field. This is more stable than raw row-level printing and gives Gold a short list of useful feature families."""),
        code("""contrast_features = [
    "grid_pos", "avg_lap", "best_lap", "std_lap", "laps_completed",
    "track_temperature", "rainfall", "stint_count", "overtakes_made",
    "avg_speed", "avg_throttle", "drs_rate"
]
contrast_features = [column for column in contrast_features if column in master.columns]
contrast_rows = []
for feature in contrast_features:
    values = pd.to_numeric(master[feature], errors="coerce")
    contrast_rows.append({
        "feature": feature,
        "winner_median": values[master["target_win"] == 1].median(),
        "non_winner_median": values[master["target_win"] == 0].median(),
        "podium_median": values[master["target_podium"] == 1].median(),
        "non_podium_median": values[master["target_podium"] == 0].median(),
        "top10_median": values[master["target_top10"] == 1].median(),
        "outside_top10_median": values[master["target_top10"] == 0].median(),
    })
contrast_df = pd.DataFrame(contrast_rows)
contrast_df["winner_delta"] = contrast_df["winner_median"] - contrast_df["non_winner_median"]
contrast_df["podium_delta"] = contrast_df["podium_median"] - contrast_df["non_podium_median"]
contrast_df["top10_delta"] = contrast_df["top10_median"] - contrast_df["outside_top10_median"]
contrast_df.to_csv(OUTPUT_TABLES / "winner_podium_top10_feature_contrast.csv", index=False)
display(contrast_df.round(3))"""),
        code("""contrast_plot = contrast_df[["feature", "winner_delta", "podium_delta", "top10_delta"]].melt(
    id_vars="feature",
    var_name="comparison",
    value_name="median_delta",
)
fig = px.bar(
    contrast_plot,
    x="median_delta",
    y="feature",
    color="comparison",
    barmode="group",
    orientation="h",
    title="Median Feature Delta by Result Class",
)
fig.write_html(OUTPUT_CHARTS / "result_class_feature_contrast.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## 4. Multicollinearity and Leakage Review

Silver should not silently promote target leakage into Gold. This section flags high feature-feature correlations and columns that are outcomes rather than pre-race or in-race explanatory signals."""),
        code("""feature_corr = numeric[available_features].corr()
high_corr_pairs = []
for i, left in enumerate(feature_corr.columns):
    for right in feature_corr.columns[i + 1:]:
        value = feature_corr.loc[left, right]
        if pd.notna(value) and abs(value) >= 0.70:
            high_corr_pairs.append({"feature_1": left, "feature_2": right, "correlation": value})
high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False) if high_corr_pairs else pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
high_corr_df.to_csv(OUTPUT_TABLES / "high_correlation_pairs.csv", index=False)
display(high_corr_df.head(30))"""),
        code("""known_leakage_columns = {
    "finish_pos": "target outcome",
    "points": "post-race scoring outcome",
    "positions_gained": "uses finish_pos and grid_pos",
    "points_per_lap": "uses post-race points",
    "target_win": "label",
    "target_podium": "label",
    "target_top10": "label",
}
leakage_review = pd.DataFrame([
    {
        "column": column,
        "reason": reason,
        "gold_policy": "exclude_from_model_features",
        "allowed_for_eda": True,
    }
    for column, reason in known_leakage_columns.items()
    if column in master.columns
])
leakage_review.to_csv(OUTPUT_TABLES / "leakage_review.csv", index=False)
display(leakage_review)"""),
        md("""## 5. Feature Selection Recommendations

The final recommendation table separates safe candidate features from EDA-only outcome columns. Gold feature engineering should prefer features that are available before or during the prediction timestamp and should avoid direct outcome-derived columns."""),
        code("""finish_corr_map = finish_corr.set_index("feature")["correlation_with_finish_pos"].to_dict()
safe_feature_candidates = [
    feature for feature in available_features
    if feature not in {"positions_gained", "points_per_lap"}
]
recommendations = []
for feature in safe_feature_candidates:
    corr_value = finish_corr_map.get(feature, np.nan)
    abs_corr = abs(corr_value) if pd.notna(corr_value) else np.nan
    if feature in {"grid_pos", "avg_lap", "median_lap", "best_lap", "std_lap", "laps_completed"}:
        family = "race_pace"
    elif feature in {"air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall"}:
        family = "weather_context"
    elif feature in {"stint_count", "max_tyre_age_start", "avg_tyre_age_start"}:
        family = "strategy_tyre"
    elif feature in {"avg_speed", "max_speed", "std_speed", "avg_rpm", "avg_throttle", "avg_brake", "drs_rate"}:
        family = "sampled_telemetry"
    else:
        family = "racecraft"
    recommendations.append({
        "feature": feature,
        "family": family,
        "correlation_with_finish_pos": corr_value,
        "abs_correlation": abs_corr,
        "recommendation": "KEEP" if pd.notna(abs_corr) and abs_corr >= 0.10 else "REVIEW",
    })
feature_recommendations = pd.DataFrame(recommendations).sort_values("abs_correlation", ascending=False)
feature_recommendations.to_csv(OUTPUT_TABLES / "gold_feature_recommendations.csv", index=False)
display(feature_recommendations.head(30))"""),
        code("""fig = px.bar(
    feature_recommendations.head(25).sort_values("abs_correlation"),
    x="abs_correlation",
    y="feature",
    color="family",
    orientation="h",
    title="Gold Candidate Feature Strength by Family",
)
fig.write_html(OUTPUT_CHARTS / "gold_feature_recommendations.html", include_plotlyjs="cdn")
fig.show()"""),
        md("""## Final Silver EDA Gate

The Silver EDA layer is complete when all seven notebooks have produced reports, a compact feature matrix exists, and leakage-sensitive columns are documented before Gold engineering begins."""),
        code("""safe_finish_corr = finish_corr[finish_corr["feature"].isin(safe_feature_candidates)].copy()
top_finish_feature = safe_finish_corr.iloc[safe_finish_corr["correlation_with_finish_pos"].abs().argmax()] if len(safe_finish_corr) else None
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "feature_matrix_rows": int(len(master)),
    "feature_matrix_columns": int(master.shape[1]),
    "candidate_feature_count": int(len(safe_feature_candidates)),
    "high_correlation_pair_count": int(len(high_corr_df)),
    "leakage_review_columns": int(len(leakage_review)),
    "top_safe_finish_corr_feature": top_finish_feature["feature"] if top_finish_feature is not None else None,
    "top_safe_finish_corr_value": float(top_finish_feature["correlation_with_finish_pos"]) if top_finish_feature is not None else 0.0,
}
write_report("correlation_analysis", report)
write_insight(
    "Silver Correlation Analysis Insights",
    [
        f"Built a compact driver-session matrix with {report['feature_matrix_rows']:,} rows and {report['feature_matrix_columns']} columns.",
        f"Reviewed {report['candidate_feature_count']} candidate model features.",
        f"Strongest safe finish-position correlation: {report['top_safe_finish_corr_feature']} = {report['top_safe_finish_corr_value']:.3f}.",
    ],
    [
        f"{len(high_corr_df)} high-correlation feature pairs need Gold feature selection review.",
        "Outcome-derived columns such as points and positions_gained are documented as EDA-only leakage risks.",
    ],
    [
        "Use grid, lap pace, stint, weather, racecraft, and sampled telemetry families as Gold inputs.",
        "Exclude labels and post-race outcome columns from training features.",
        "Resolve high-correlation pairs before model training to reduce redundant signal.",
    ],
)
(CHECKPOINTS / "silver_eda_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)
print("Silver EDA completed. Ready for Gold feature engineering.")"""),
    ], NB_DIR / "07_correlation_analysis.ipynb")


if __name__ == "__main__":
    build()

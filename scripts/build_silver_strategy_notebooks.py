"""Build Silver race-strategy EDA notebooks.

These notebooks extend the Silver layer beyond validation into F1 strategy
analysis: pit stops, overtakes, interval gaps, and tyre stints.
"""

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


def setup(notebook_name: str) -> str:
    return f"""from pathlib import Path
import sys
from datetime import datetime
import json

import pandas as pd
import numpy as np
import plotly.express as px

ROOT = Path.cwd()
while not (ROOT / "configs" / "pipeline_config.yaml").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

SHARED = ROOT / "eda" / "shared" / "scripts"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from config import CLEANED_DATA_PATH

NOTEBOOK_NAME = "{notebook_name}"
OUTPUT_TABLES = ROOT / "eda" / "silver" / "outputs" / "tables" / NOTEBOOK_NAME
OUTPUT_CHARTS = ROOT / "eda" / "silver" / "outputs" / "charts" / NOTEBOOK_NAME
OUTPUT_REPORTS = ROOT / "eda" / "silver" / "outputs" / "reports" / NOTEBOOK_NAME
INSIGHTS = ROOT / "eda" / "silver" / "insights"
CHECKPOINTS = ROOT / "eda" / "silver" / "checkpoints"
for path in [OUTPUT_TABLES, OUTPUT_CHARTS, OUTPUT_REPORTS, INSIGHTS, CHECKPOINTS]:
    path.mkdir(parents=True, exist_ok=True)

def write_report(name: str, payload: dict) -> None:
    (OUTPUT_REPORTS / f"{{name}}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def write_insight(title: str, observations: list[str], issues: list[str], recommendations: list[str]) -> None:
    content = f"# {{title}}\\n\\n"
    content += f"**Generated at:** {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\\n\\n"
    content += "## Key Observations\\n\\n" + "\\n".join(f"- {{item}}" for item in observations) + "\\n\\n"
    content += "## Issues\\n\\n" + ("\\n".join(f"- {{item}}" for item in issues) if issues else "- None") + "\\n\\n"
    content += "## Recommendations\\n\\n" + "\\n".join(f"- {{item}}" for item in recommendations) + "\\n"
    (INSIGHTS / f"{{NOTEBOOK_NAME}}.md").write_text(content, encoding="utf-8")

def add_event_type(sessions: pd.DataFrame) -> pd.DataFrame:
    sessions = sessions.copy()
    sessions["event_type"] = np.where(
        sessions["session_name"].astype(str).str.lower().eq("sprint"),
        "SPRINT_RACE",
        "GRAND_PRIX_RACE",
    )
    return sessions

def save_fig(fig, name: str):
    fig.write_html(OUTPUT_CHARTS / f"{{name}}.html", include_plotlyjs="cdn")
    fig.show()

print("=" * 72)
print(f"SILVER STRATEGY EDA - {{NOTEBOOK_NAME}}")
print(f"Start time: {{datetime.now()}}")
print(f"Cleaned data path: {{CLEANED_DATA_PATH}}")
print("=" * 72)
"""


def build() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)

    write_notebook([
        md("""# 08 Pit Stop Analysis

Pit stops are the clearest operational expression of F1 strategy: teams trade track position for tyre performance, and the cost of that trade varies by circuit, weather, and execution quality.

This notebook reads only `data/cleaned/pit.parquet`. Silver EDA must not read Bronze/raw files, because doing so hides a broken Bronze-to-Silver contract."""),
        code(setup("08_pit_stop_analysis")),
        md("""## 1. Pit Data Coverage

Before judging strategy, we verify whether pit stops have been promoted into Silver. A missing Silver pit artifact is not a charting problem; it is a pipeline coverage problem because pit timing is central to tyre strategy features."""),
        code("""pit_path = CLEANED_DATA_PATH / "pit.parquet"
if not pit_path.exists():
    raise FileNotFoundError(
        "Missing Silver artifact: data/cleaned/pit.parquet. "
        "Run the Bronze-to-Silver cleaning step after enabling pit cleaning."
    )
pit = pd.read_parquet(pit_path)

sessions = add_event_type(pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet"))
drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
weather = pd.read_parquet(CLEANED_DATA_PATH / "weather.parquet")
starting_grid = pd.read_parquet(CLEANED_DATA_PATH / "starting_grid.parquet")

for column in ["pit_duration", "lane_duration", "stop_duration", "lap_number", "driver_number", "session_key"]:
    if column in pit.columns:
        pit[column] = pd.to_numeric(pit[column], errors="coerce")
if "date" in pit.columns:
    pit["date"] = pd.to_datetime(pit["date"], errors="coerce", utc=True)

coverage = pd.DataFrame([{
    "pit_source": "silver",
    "silver_pit_exists": pit_path.exists(),
    "rows": len(pit),
    "sessions": pit["session_key"].nunique() if "session_key" in pit.columns and len(pit) else 0,
    "drivers": pit["driver_number"].nunique() if "driver_number" in pit.columns and len(pit) else 0,
    "coverage_issue": None,
}])
coverage.to_csv(OUTPUT_TABLES / "pit_coverage.csv", index=False)
display(coverage)"""),
        code("""fig = px.bar(
    coverage,
    x="pit_source",
    y="rows",
    color="silver_pit_exists",
    title="Pit Stop Coverage Source",
    text="rows",
)
save_fig(fig, "pit_coverage")"""),
        md("""## 2. Pit Duration Distribution

Pit duration combines lane loss and stationary servicing behavior depending on source semantics. We therefore profile `pit_duration`, `lane_duration`, and `stop_duration` separately when present."""),
        code("""duration_cols = [column for column in ["pit_duration", "lane_duration", "stop_duration"] if column in pit.columns]
duration_summary = pit[duration_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).reset_index().rename(columns={"index": "metric"}) if duration_cols else pd.DataFrame()
duration_summary.to_csv(OUTPUT_TABLES / "pit_duration_summary.csv", index=False)
display(duration_summary)

pit_long = pit[duration_cols].melt(var_name="duration_type", value_name="seconds").dropna() if duration_cols else pd.DataFrame()
pit_long = pit_long[pit_long["seconds"].between(0, 180)]
fig = px.histogram(
    pit_long,
    x="seconds",
    color="duration_type",
    nbins=60,
    barmode="overlay",
    title="Pit Stop Duration Distribution",
)
save_fig(fig, "pit_duration_distribution")"""),
        md("""## 3. Pit Windows and Team Execution

The lap of the first stop often reveals the tyre strategy window. Team ranking is joined on `session_key + driver_number` to avoid mixing drivers across sessions when seat/team changes occur."""),
        code("""pit_enriched = pit.merge(
    drivers[["session_key", "driver_number", "full_name", "team_name"]].drop_duplicates(["session_key", "driver_number"]),
    on=["session_key", "driver_number"],
    how="left",
)
pit_enriched = pit_enriched.merge(
    sessions[["session_key", "year", "event_type", "circuit_short_name", "country_name"]],
    on="session_key",
    how="left",
)
pit_enriched["valid_pit_duration"] = pit_enriched["pit_duration"].where(pit_enriched["pit_duration"].between(5, 120)) if "pit_duration" in pit_enriched.columns else np.nan

pit_window = pit_enriched.dropna(subset=["lap_number"]).groupby(["event_type", "lap_number"], as_index=False).size().rename(columns={"size": "pit_stops"})
pit_window.to_csv(OUTPUT_TABLES / "pit_window_by_lap.csv", index=False)
display(pit_window.sort_values("pit_stops", ascending=False).head(15))

fig = px.line(
    pit_window,
    x="lap_number",
    y="pit_stops",
    color="event_type",
    title="Pit Stop Timing Window by Lap",
)
save_fig(fig, "pit_window_by_lap")"""),
        code("""team_pit = pit_enriched.dropna(subset=["valid_pit_duration", "team_name"]).groupby("team_name", as_index=False).agg(
    avg_pit_duration=("valid_pit_duration", "mean"),
    median_pit_duration=("valid_pit_duration", "median"),
    pit_count=("valid_pit_duration", "count"),
    p95_pit_duration=("valid_pit_duration", lambda s: s.quantile(0.95)),
)
team_pit = team_pit[team_pit["pit_count"] >= 10].sort_values("median_pit_duration")
team_pit.to_csv(OUTPUT_TABLES / "team_pit_performance.csv", index=False)
display(team_pit)

fig = px.bar(
    team_pit,
    x="median_pit_duration",
    y="team_name",
    orientation="h",
    color="pit_count",
    title="Team Pit Execution Ranking",
)
save_fig(fig, "team_pit_performance")"""),
        md("""## 4. Strategic Impact

Pit performance alone rarely explains the race result. We combine pit count, average pit duration, grid position, finish position, and weather to separate operational execution from strategic context."""),
        code("""driver_pit = pit_enriched.groupby(["session_key", "driver_number"], as_index=False).agg(
    pit_count=("lap_number", "count"),
    avg_pit_duration=("valid_pit_duration", "mean"),
    first_pit_lap=("lap_number", "min"),
    last_pit_lap=("lap_number", "max"),
)
finish = session_result[["session_key", "driver_number", "position", "points", "dnf", "dns", "dsq"]].copy()
finish["finish_pos"] = pd.to_numeric(finish["position"], errors="coerce")
grid = starting_grid[["session_key", "driver_number", "position"]].rename(columns={"position": "grid_pos"})
grid["grid_pos"] = pd.to_numeric(grid["grid_pos"], errors="coerce")
session_weather = weather.groupby("session_key", as_index=False).agg(
    rainfall=("rainfall", "max"),
    track_temperature=("track_temperature", "mean"),
)
pit_impact = (
    driver_pit
    .merge(finish, on=["session_key", "driver_number"], how="left")
    .merge(grid, on=["session_key", "driver_number"], how="left")
    .merge(session_weather, on="session_key", how="left")
)
pit_impact["positions_gained"] = pit_impact["grid_pos"] - pit_impact["finish_pos"]
pit_impact.to_csv(OUTPUT_TABLES / "pit_strategy_impact.csv", index=False)
display(pit_impact.head(20))

impact_corr = pit_impact[["pit_count", "avg_pit_duration", "first_pit_lap", "positions_gained", "finish_pos", "track_temperature", "rainfall"]].corr()
impact_corr.to_csv(OUTPUT_TABLES / "pit_impact_correlations.csv")
fig = px.imshow(
    impact_corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Pit Strategy Impact Correlation Matrix",
)
save_fig(fig, "pit_strategy_impact_correlation")"""),
        code("""weather_pit = pit_enriched.merge(session_weather, on="session_key", how="left")
weather_pit["condition"] = np.where(weather_pit["rainfall"].fillna(0) > 0, "Wet session", "Dry session")
weather_summary = weather_pit.dropna(subset=["valid_pit_duration"]).groupby("condition", as_index=False).agg(
    pit_stops=("valid_pit_duration", "count"),
    avg_pit_duration=("valid_pit_duration", "mean"),
    median_pit_duration=("valid_pit_duration", "median"),
)
weather_summary.to_csv(OUTPUT_TABLES / "pit_weather_summary.csv", index=False)
display(weather_summary)

fig = px.box(
    weather_pit.dropna(subset=["valid_pit_duration"]),
    x="condition",
    y="valid_pit_duration",
    color="condition",
    title="Pit Duration Under Dry vs Wet Sessions",
)
save_fig(fig, "pit_weather_impact")"""),
        md("""## Final Pit Stop Report"""),
        code("""fastest_team = team_pit.iloc[0].to_dict() if len(team_pit) else {}
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "pit_source": "silver",
    "pit_rows": int(len(pit)),
    "sessions_with_pit": int(pit["session_key"].nunique()) if "session_key" in pit.columns and len(pit) else 0,
    "median_pit_duration": float(pit_enriched["valid_pit_duration"].median()) if "valid_pit_duration" in pit_enriched.columns else 0.0,
    "fastest_team": fastest_team.get("team_name"),
    "fastest_team_median_duration": float(fastest_team.get("median_pit_duration", 0.0)) if fastest_team else 0.0,
    "coverage_issue": None,
}
write_report("pit_stop_analysis", report)
write_insight(
    "Silver Pit Stop Analysis Insights",
    [
        f"Analyzed {report['pit_rows']:,} pit records from {report['pit_source']}.",
        f"Median valid pit duration: {report['median_pit_duration']:.2f}s.",
        f"Fastest team by median duration: {report['fastest_team']}.",
    ],
    [],
    [
        "Keep pit as a required Silver artifact; never fallback to raw files inside Silver EDA.",
        "Use first_pit_lap, pit_count, and team rolling pit duration as Gold strategy features.",
        "Treat wet sessions separately because pit timing and tyre calls change under rain risk.",
    ],
)
(CHECKPOINTS / "silver_pit_stop_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "08_pit_stop_analysis.ipynb")

    write_notebook([
        md("""# 09 Overtake Analysis

Overtaking captures racecraft, circuit passability, and traffic dynamics. This notebook analyzes who creates overtakes, who loses positions, where overtaking clusters, and whether overtaking signal aligns with finish position."""),
        code(setup("09_overtake_analysis")),
        code("""overtakes = pd.read_parquet(CLEANED_DATA_PATH / "overtakes.parquet")
drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
sessions = add_event_type(pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet"))
session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")

for column in ["overtaking_driver_number", "overtaken_driver_number", "position", "session_key"]:
    if column in overtakes.columns:
        overtakes[column] = pd.to_numeric(overtakes[column], errors="coerce")
overtakes["date"] = pd.to_datetime(overtakes["date"], errors="coerce", utc=True)
overtakes = overtakes.merge(sessions[["session_key", "year", "event_type", "circuit_short_name", "date_start", "date_end"]], on="session_key", how="left")
for column in ["date_start", "date_end"]:
    overtakes[column] = pd.to_datetime(overtakes[column], errors="coerce", utc=True)
duration_seconds = (overtakes["date_end"] - overtakes["date_start"]).dt.total_seconds()
overtakes["race_phase_pct"] = ((overtakes["date"] - overtakes["date_start"]).dt.total_seconds() / duration_seconds * 100).clip(0, 100)
overtakes["race_phase"] = pd.cut(overtakes["race_phase_pct"], bins=[-1, 25, 50, 75, 101], labels=["Start", "Early-Mid", "Late-Mid", "Final"])

coverage = pd.DataFrame([{
    "rows": len(overtakes),
    "sessions": overtakes["session_key"].nunique(),
    "circuits": overtakes["circuit_short_name"].nunique(),
    "drivers_making_overtakes": overtakes["overtaking_driver_number"].nunique(),
}])
coverage.to_csv(OUTPUT_TABLES / "overtake_coverage.csv", index=False)
display(coverage)"""),
        md("""## 1. Overtake Volume and Timing

Overtake count by session tells us which races are strategically alive versus processional. Race phase helps identify whether passing happens mostly during opening chaos, tyre-offset windows, or late-race recovery."""),
        code("""session_overtakes = overtakes.groupby(["session_key", "year", "event_type", "circuit_short_name"], as_index=False).size().rename(columns={"size": "overtakes"})
session_overtakes.to_csv(OUTPUT_TABLES / "session_overtake_counts.csv", index=False)
display(session_overtakes.sort_values("overtakes", ascending=False).head(20))

fig = px.histogram(session_overtakes, x="overtakes", color="event_type", nbins=30, title="Overtakes per Session Distribution")
save_fig(fig, "overtakes_per_session")"""),
        code("""phase_summary = overtakes.groupby(["event_type", "race_phase"], observed=True, as_index=False).size().rename(columns={"size": "overtakes"})
phase_summary.to_csv(OUTPUT_TABLES / "overtake_phase_summary.csv", index=False)
display(phase_summary)

fig = px.bar(
    phase_summary,
    x="race_phase",
    y="overtakes",
    color="event_type",
    barmode="group",
    title="Overtake Timing by Race Phase",
)
save_fig(fig, "overtake_timing_phase")"""),
        md("""## 2. Driver Racecraft Balance

Net overtakes separate aggressive progress from vulnerability in traffic. The join uses session-aware driver identity to avoid name collisions across seasons and teams."""),
        code("""driver_dim = drivers[["session_key", "driver_number", "full_name", "team_name"]].drop_duplicates(["session_key", "driver_number"])
made = overtakes.groupby(["session_key", "overtaking_driver_number"]).size().reset_index(name="overtakes_made").rename(columns={"overtaking_driver_number": "driver_number"})
lost = overtakes.groupby(["session_key", "overtaken_driver_number"]).size().reset_index(name="overtaken_count").rename(columns={"overtaken_driver_number": "driver_number"})
driver_racecraft = made.merge(lost, on=["session_key", "driver_number"], how="outer").fillna(0)
driver_racecraft = driver_racecraft.merge(driver_dim, on=["session_key", "driver_number"], how="left")
driver_racecraft["driver_id"] = driver_racecraft["full_name"].fillna("Driver " + driver_racecraft["driver_number"].astype(int).astype(str))
driver_racecraft["net_overtakes"] = driver_racecraft["overtakes_made"] - driver_racecraft["overtaken_count"]
driver_summary = driver_racecraft.groupby("driver_id", as_index=False).agg(
    overtakes_made=("overtakes_made", "sum"),
    overtaken_count=("overtaken_count", "sum"),
    net_overtakes=("net_overtakes", "sum"),
    sessions=("session_key", "nunique"),
)
driver_summary["net_per_session"] = driver_summary["net_overtakes"] / driver_summary["sessions"]
driver_summary = driver_summary.sort_values("net_overtakes", ascending=False)
driver_summary.to_csv(OUTPUT_TABLES / "driver_overtake_balance.csv", index=False)
display(driver_summary.head(20))"""),
        code("""fig = px.bar(
    driver_summary.head(20).sort_values("net_overtakes"),
    x="net_overtakes",
    y="driver_id",
    orientation="h",
    color="net_per_session",
    title="Driver Net Overtake Balance",
)
save_fig(fig, "driver_net_overtakes")"""),
        md("""## 3. Circuit Passability and Result Signal

Circuit ranking identifies tracks where traffic can be solved on track versus tracks where qualifying and pit strategy become more decisive."""),
        code("""circuit_overtakes = session_overtakes.groupby("circuit_short_name", as_index=False).agg(
    sessions=("session_key", "nunique"),
    total_overtakes=("overtakes", "sum"),
    avg_overtakes=("overtakes", "mean"),
).sort_values("avg_overtakes", ascending=False)
circuit_overtakes.to_csv(OUTPUT_TABLES / "circuit_overtake_ranking.csv", index=False)
display(circuit_overtakes)

fig = px.bar(
    circuit_overtakes,
    x="avg_overtakes",
    y="circuit_short_name",
    orientation="h",
    color="sessions",
    title="Circuit Overtaking Ranking",
)
save_fig(fig, "circuit_overtake_ranking")"""),
        code("""finish = session_result[["session_key", "driver_number", "position", "points"]].copy()
finish["finish_pos"] = pd.to_numeric(finish["position"], errors="coerce")
overtake_finish = driver_racecraft.merge(finish, on=["session_key", "driver_number"], how="left")
overtake_corr = overtake_finish[["overtakes_made", "overtaken_count", "net_overtakes", "finish_pos", "points"]].corr()
overtake_corr.to_csv(OUTPUT_TABLES / "overtake_finish_correlations.csv")
display(overtake_corr.round(3))

fig = px.imshow(
    overtake_corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Overtaking vs Race Outcome Correlations",
)
save_fig(fig, "overtake_finish_correlation")"""),
        md("""## Final Overtake Report"""),
        code("""top_driver = driver_summary.iloc[0].to_dict() if len(driver_summary) else {}
top_circuit = circuit_overtakes.iloc[0].to_dict() if len(circuit_overtakes) else {}
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "overtake_rows": int(len(overtakes)),
    "sessions": int(overtakes["session_key"].nunique()),
    "top_net_overtake_driver": top_driver.get("driver_id"),
    "top_net_overtakes": float(top_driver.get("net_overtakes", 0)),
    "highest_overtake_circuit": top_circuit.get("circuit_short_name"),
    "highest_avg_overtakes": float(top_circuit.get("avg_overtakes", 0)),
}
write_report("overtake_analysis", report)
write_insight(
    "Silver Overtake Analysis Insights",
    [
        f"Analyzed {report['overtake_rows']:,} overtake events across {report['sessions']} sessions.",
        f"Top net overtake driver: {report['top_net_overtake_driver']} ({report['top_net_overtakes']:.0f}).",
        f"Highest average overtake circuit: {report['highest_overtake_circuit']}.",
    ],
    [],
    [
        "Use net overtakes and circuit passability as Gold racecraft features.",
        "Normalize overtake volume by event type because Sprint and Grand Prix sessions have different opportunity lengths.",
        "Use race_phase_pct for future strategy timing features.",
    ],
)
(CHECKPOINTS / "silver_overtake_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "09_overtake_analysis.ipynb")

    write_notebook([
        md("""# 10 Interval Analysis

Intervals describe race compression: whether cars are bunched in DRS trains, spread into clean-air gaps, or reset by interruptions. This notebook profiles gap dynamics without reading telemetry."""),
        code(setup("10_interval_analysis")),
        code("""intervals = pd.read_parquet(CLEANED_DATA_PATH / "intervals.parquet")
sessions = add_event_type(pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet"))
drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")
intervals["date"] = pd.to_datetime(intervals["date"], errors="coerce", utc=True)
for column in ["interval", "gap_to_leader", "driver_number", "session_key"]:
    if column in intervals.columns:
        intervals[column] = pd.to_numeric(intervals[column], errors="coerce")
intervals = intervals.merge(sessions[["session_key", "year", "event_type", "circuit_short_name", "date_start", "date_end"]], on="session_key", how="left")
intervals["date_start"] = pd.to_datetime(intervals["date_start"], errors="coerce", utc=True)
intervals["date_end"] = pd.to_datetime(intervals["date_end"], errors="coerce", utc=True)
duration_seconds = (intervals["date_end"] - intervals["date_start"]).dt.total_seconds()
intervals["race_phase_pct"] = ((intervals["date"] - intervals["date_start"]).dt.total_seconds() / duration_seconds * 100).clip(0, 100)

coverage = pd.DataFrame([{
    "rows": len(intervals),
    "sessions": intervals["session_key"].nunique(),
    "circuits": intervals["circuit_short_name"].nunique(),
    "valid_interval_rows": int(intervals["interval"].notna().sum()),
}])
coverage.to_csv(OUTPUT_TABLES / "interval_coverage.csv", index=False)
display(coverage)"""),
        md("""## 1. Gap Distribution

The interval distribution separates DRS-range battles from clean-air separation. Values between roughly 0.5s and 1.5s are treated as potential DRS-train conditions, not exact FIA DRS eligibility."""),
        code("""valid_intervals = intervals[intervals["interval"].between(0, 30)].copy()
interval_stats = valid_intervals["interval"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).reset_index()
interval_stats.columns = ["metric", "interval_seconds"]
interval_stats.to_csv(OUTPUT_TABLES / "interval_distribution_stats.csv", index=False)
display(interval_stats)

fig = px.histogram(
    valid_intervals.sample(min(200_000, len(valid_intervals)), random_state=42),
    x="interval",
    color="event_type",
    nbins=80,
    title="Car-to-Car Interval Distribution",
)
save_fig(fig, "interval_distribution")"""),
        code("""phase_gap = valid_intervals.groupby(["session_key", "event_type", "circuit_short_name"], as_index=False).agg(
    median_interval=("interval", "median"),
    p25_interval=("interval", lambda s: s.quantile(0.25)),
    drs_train_rate=("interval", lambda s: s.between(0.5, 1.5).mean()),
    compressed_rate=("interval", lambda s: s.le(1.5).mean()),
    observations=("interval", "count"),
)
phase_gap.to_csv(OUTPUT_TABLES / "session_gap_compression.csv", index=False)
display(phase_gap.sort_values("drs_train_rate", ascending=False).head(20))

fig = px.scatter(
    phase_gap,
    x="median_interval",
    y="drs_train_rate",
    size="observations",
    color="event_type",
    hover_name="circuit_short_name",
    title="Race Compression: Median Interval vs DRS-Train Rate",
)
save_fig(fig, "race_compression_scatter")"""),
        md("""## 2. When Gaps Form

Since intervals are timestamped rather than lap-numbered, we use normalized race phase to estimate whether fields spread early or remain compressed deep into the session."""),
        code("""valid_intervals = valid_intervals.copy()
valid_intervals["phase_bucket"] = pd.cut(valid_intervals["race_phase_pct"], bins=[-1, 20, 40, 60, 80, 101], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
phase_compression = valid_intervals.groupby(["event_type", "phase_bucket"], observed=True, as_index=False).agg(
    median_interval=("interval", "median"),
    drs_train_rate=("interval", lambda s: s.between(0.5, 1.5).mean()),
    observations=("interval", "count"),
)
phase_compression.to_csv(OUTPUT_TABLES / "interval_phase_compression.csv", index=False)
display(phase_compression)

fig = px.line(
    phase_compression,
    x="phase_bucket",
    y="median_interval",
    color="event_type",
    markers=True,
    title="Gap Formation Across Race Phase",
)
save_fig(fig, "gap_formation_by_phase")"""),
        md("""## 3. Race Classification

Sessions are classified as compressed, balanced, or processional using median interval and DRS-train rate. This is an analytical label for feature design, not an official race classification."""),
        code("""def classify_race(row):
    if row["drs_train_rate"] >= 0.20:
        return "DRS_TRAIN_HEAVY"
    if row["median_interval"] >= 3.0:
        return "PROCESSIONAL_SPREAD"
    return "BALANCED_RACE"

race_class = phase_gap.copy()
race_class["race_gap_class"] = race_class.apply(classify_race, axis=1)
race_class.to_csv(OUTPUT_TABLES / "race_gap_classification.csv", index=False)
display(race_class["race_gap_class"].value_counts().reset_index().rename(columns={"index": "race_gap_class", "race_gap_class": "sessions"}))

fig = px.bar(
    race_class["race_gap_class"].value_counts().reset_index(),
    x="race_gap_class",
    y="count",
    title="Race Gap Classification",
)
save_fig(fig, "race_gap_classification")"""),
        code("""circuit_gap = race_class.groupby("circuit_short_name", as_index=False).agg(
    sessions=("session_key", "nunique"),
    avg_median_interval=("median_interval", "mean"),
    avg_drs_train_rate=("drs_train_rate", "mean"),
    avg_compressed_rate=("compressed_rate", "mean"),
).sort_values("avg_drs_train_rate", ascending=False)
circuit_gap.to_csv(OUTPUT_TABLES / "circuit_gap_profiles.csv", index=False)
display(circuit_gap)

fig = px.bar(
    circuit_gap,
    x="avg_drs_train_rate",
    y="circuit_short_name",
    orientation="h",
    color="avg_median_interval",
    title="Circuit DRS-Train Profile",
)
save_fig(fig, "circuit_drs_train_profile")"""),
        md("""## Final Interval Report"""),
        code("""top_train = circuit_gap.iloc[0].to_dict() if len(circuit_gap) else {}
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "interval_rows": int(len(intervals)),
    "valid_interval_rows": int(len(valid_intervals)),
    "sessions": int(intervals["session_key"].nunique()),
    "avg_drs_train_rate": float(race_class["drs_train_rate"].mean()) if len(race_class) else 0.0,
    "highest_drs_train_circuit": top_train.get("circuit_short_name"),
    "highest_drs_train_rate": float(top_train.get("avg_drs_train_rate", 0)),
}
write_report("interval_analysis", report)
write_insight(
    "Silver Interval Analysis Insights",
    [
        f"Analyzed {report['valid_interval_rows']:,} valid interval observations.",
        f"Average session DRS-train rate: {report['avg_drs_train_rate']:.3f}.",
        f"Highest DRS-train circuit: {report['highest_drs_train_circuit']}.",
    ],
    [],
    [
        "Use session-level gap class as a Gold race-context feature.",
        "Combine DRS-train rate with overtaking volume to distinguish passable tracks from traffic traps.",
        "Use normalized race phase for interval features because raw interval data is timestamped rather than lap-indexed.",
    ],
)
(CHECKPOINTS / "silver_interval_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "10_interval_analysis.ipynb")

    write_notebook([
        md("""# 11 Stint Analysis

Stints connect tyre compound, tyre age, lap duration, and pit strategy. This notebook turns Silver stints into tyre degradation and strategy signals for Gold feature engineering."""),
        code(setup("11_stint_analysis")),
        code("""stints = pd.read_parquet(CLEANED_DATA_PATH / "stints.parquet")
laps = pd.read_parquet(CLEANED_DATA_PATH / "laps.parquet")
drivers = pd.read_parquet(CLEANED_DATA_PATH / "drivers.parquet")
sessions = add_event_type(pd.read_parquet(CLEANED_DATA_PATH / "sessions.parquet"))
session_result = pd.read_parquet(CLEANED_DATA_PATH / "session_result.parquet")

for column in ["lap_start", "lap_end", "tyre_age_at_start", "stint_number", "driver_number", "session_key"]:
    if column in stints.columns:
        stints[column] = pd.to_numeric(stints[column], errors="coerce")
stints["stint_length"] = stints["lap_end"] - stints["lap_start"] + 1
stints = stints.merge(sessions[["session_key", "year", "event_type", "circuit_short_name"]], on="session_key", how="left")
stints = stints.merge(drivers[["session_key", "driver_number", "full_name", "team_name"]].drop_duplicates(["session_key", "driver_number"]), on=["session_key", "driver_number"], how="left")
stints["driver_id"] = stints["full_name"].fillna("Driver " + stints["driver_number"].astype(int).astype(str))

coverage = pd.DataFrame([{
    "stint_rows": len(stints),
    "sessions": stints["session_key"].nunique(),
    "drivers": stints["driver_number"].nunique(),
    "compounds": stints["compound"].nunique(),
}])
coverage.to_csv(OUTPUT_TABLES / "stint_coverage.csv", index=False)
display(coverage)"""),
        md("""## 1. Stint Length and Compound Mix

Compound mix explains how teams trade pace against durability. Stint length by compound gives the first approximation of tyre life before modeling lap-level degradation."""),
        code("""stint_summary = stints.groupby("compound", as_index=False).agg(
    stints=("stint_number", "count"),
    avg_stint_length=("stint_length", "mean"),
    median_stint_length=("stint_length", "median"),
    p90_stint_length=("stint_length", lambda s: s.quantile(0.90)),
    avg_tyre_age_start=("tyre_age_at_start", "mean"),
).sort_values("stints", ascending=False)
stint_summary.to_csv(OUTPUT_TABLES / "compound_stint_summary.csv", index=False)
display(stint_summary)

fig = px.bar(
    stint_summary,
    x="compound",
    y="stints",
    color="median_stint_length",
    title="Tyre Compound Usage and Median Stint Length",
)
save_fig(fig, "compound_usage")"""),
        code("""fig = px.box(
    stints.dropna(subset=["stint_length", "compound"]),
    x="compound",
    y="stint_length",
    color="compound",
    title="Stint Length Distribution by Compound",
)
save_fig(fig, "stint_length_by_compound")"""),
        md("""## 2. Lap-Time Degradation by Compound

The lap table is joined to stint ranges to estimate tyre age within a stint. This is intentionally aggregated before plotting so the notebook remains light."""),
        code("""lap_work = laps[["session_key", "driver_number", "lap_number", "lap_duration", "is_pit_out_lap"]].copy()
lap_work["lap_number"] = pd.to_numeric(lap_work["lap_number"], errors="coerce")
lap_work["lap_duration"] = pd.to_numeric(lap_work["lap_duration"], errors="coerce")
lap_work = lap_work[lap_work["lap_duration"].between(50, 900)]

stint_ranges = stints[["session_key", "driver_number", "stint_number", "compound", "lap_start", "lap_end", "tyre_age_at_start"]].dropna(subset=["lap_start", "lap_end"])
lap_stints = lap_work.merge(stint_ranges, on=["session_key", "driver_number"], how="inner")
lap_stints = lap_stints[(lap_stints["lap_number"] >= lap_stints["lap_start"]) & (lap_stints["lap_number"] <= lap_stints["lap_end"])]
lap_stints["stint_lap_index"] = lap_stints["lap_number"] - lap_stints["lap_start"] + 1
lap_stints["estimated_tyre_age"] = lap_stints["tyre_age_at_start"] + lap_stints["stint_lap_index"] - 1
degradation = lap_stints.groupby(["compound", "stint_lap_index"], as_index=False).agg(
    median_lap_duration=("lap_duration", "median"),
    avg_lap_duration=("lap_duration", "mean"),
    laps=("lap_duration", "count"),
)
degradation = degradation[degradation["laps"] >= 20]
degradation.to_csv(OUTPUT_TABLES / "compound_lap_degradation.csv", index=False)
display(degradation.head(30))

fig = px.line(
    degradation,
    x="stint_lap_index",
    y="median_lap_duration",
    color="compound",
    markers=True,
    title="Tyre Degradation Curve by Compound",
)
save_fig(fig, "compound_degradation_curve")"""),
        md("""## 3. Driver and Team Stint Management

Long stints are not automatically good; they must be interpreted with lap-time stability. We summarize who extends stints while keeping lap-time variance controlled."""),
        code("""driver_stint = lap_stints.groupby(["driver_number", "compound", "session_key", "stint_number"], as_index=False).agg(
    stint_laps=("lap_number", "count"),
    median_lap=("lap_duration", "median"),
    lap_std=("lap_duration", "std"),
)
driver_stint = driver_stint.merge(stints[["session_key", "driver_number", "stint_number", "driver_id", "team_name"]], on=["session_key", "driver_number", "stint_number"], how="left")
driver_management = driver_stint.groupby("driver_id", as_index=False).agg(
    avg_stint_laps=("stint_laps", "mean"),
    max_stint_laps=("stint_laps", "max"),
    avg_lap_std=("lap_std", "mean"),
    stints=("stint_number", "count"),
)
driver_management = driver_management[driver_management["stints"] >= 10].sort_values(["avg_stint_laps", "avg_lap_std"], ascending=[False, True])
driver_management.to_csv(OUTPUT_TABLES / "driver_stint_management.csv", index=False)
display(driver_management.head(20))

fig = px.scatter(
    driver_management,
    x="avg_stint_laps",
    y="avg_lap_std",
    size="stints",
    hover_name="driver_id",
    title="Driver Stint Extension vs Lap-Time Variability",
)
save_fig(fig, "driver_stint_management")"""),
        code("""team_strategy = stints.groupby(["session_key", "team_name", "driver_number"], as_index=False).agg(
    stint_count=("stint_number", "nunique"),
    compounds_used=("compound", "nunique"),
    max_stint_length=("stint_length", "max"),
)
team_strategy_summary = team_strategy.groupby("team_name", as_index=False).agg(
    avg_stints=("stint_count", "mean"),
    one_stop_rate=("stint_count", lambda s: (s == 2).mean()),
    two_stop_rate=("stint_count", lambda s: (s == 3).mean()),
    three_plus_stop_rate=("stint_count", lambda s: (s >= 4).mean()),
    avg_compounds_used=("compounds_used", "mean"),
)
team_strategy_summary.to_csv(OUTPUT_TABLES / "team_strategy_patterns.csv", index=False)
display(team_strategy_summary.sort_values("avg_stints", ascending=False))

fig = px.bar(
    team_strategy_summary.sort_values("avg_stints"),
    x="avg_stints",
    y="team_name",
    orientation="h",
    color="avg_compounds_used",
    title="Team Strategy Pattern: Average Stints and Compound Variety",
)
save_fig(fig, "team_strategy_patterns")"""),
        md("""## 4. Undercut Proxy and Used-Tyre Signal

A full undercut model needs exact pit and position timing. As a Silver proxy, we compare median lap time in the final laps before a stint change with the opening flying laps after the next stint begins."""),
        code("""ordered = lap_stints.sort_values(["session_key", "driver_number", "stint_number", "stint_lap_index"])
stint_edges = []
for (session_key, driver_number), group in ordered.groupby(["session_key", "driver_number"]):
    grouped = {stint: data for stint, data in group.groupby("stint_number")}
    for stint_number in sorted(grouped):
        next_number = stint_number + 1
        if next_number not in grouped:
            continue
        before = grouped[stint_number].tail(3)["lap_duration"].median()
        after = grouped[next_number].head(3)["lap_duration"].median()
        stint_edges.append({
            "session_key": session_key,
            "driver_number": driver_number,
            "stint_number": stint_number,
            "next_stint_number": next_number,
            "pre_stop_median_lap": before,
            "post_stop_median_lap": after,
            "lap_delta_after_stop": after - before,
        })
undercut_proxy = pd.DataFrame(stint_edges)
undercut_proxy.to_csv(OUTPUT_TABLES / "undercut_proxy_lap_delta.csv", index=False)
display(undercut_proxy.describe().reset_index() if len(undercut_proxy) else undercut_proxy)

fig = px.histogram(
    undercut_proxy.dropna(subset=["lap_delta_after_stop"]),
    x="lap_delta_after_stop",
    nbins=60,
    title="Post-Stop Lap Delta Proxy",
)
save_fig(fig, "undercut_proxy_lap_delta")"""),
        code("""stints["tyre_start_class"] = np.where(stints["tyre_age_at_start"].fillna(0) <= 1, "New/near-new", "Used")
used_tyre_summary = stints.groupby(["compound", "tyre_start_class"], as_index=False).agg(
    stints=("stint_number", "count"),
    avg_stint_length=("stint_length", "mean"),
    median_stint_length=("stint_length", "median"),
)
used_tyre_summary.to_csv(OUTPUT_TABLES / "used_tyre_strategy.csv", index=False)
display(used_tyre_summary)

fig = px.bar(
    used_tyre_summary,
    x="compound",
    y="median_stint_length",
    color="tyre_start_class",
    barmode="group",
    title="New vs Used Tyre Stint Length",
)
save_fig(fig, "used_tyre_strategy")"""),
        md("""## Final Stint Report"""),
        code("""top_compound = stint_summary.iloc[0].to_dict() if len(stint_summary) else {}
top_manager = driver_management.iloc[0].to_dict() if len(driver_management) else {}
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "status": "PASS",
    "stint_rows": int(len(stints)),
    "lap_stint_rows": int(len(lap_stints)),
    "most_used_compound": top_compound.get("compound"),
    "most_used_compound_stints": int(top_compound.get("stints", 0)) if top_compound else 0,
    "top_stint_extension_driver": top_manager.get("driver_id"),
    "top_driver_avg_stint_laps": float(top_manager.get("avg_stint_laps", 0)) if top_manager else 0.0,
}
write_report("stint_analysis", report)
write_insight(
    "Silver Stint Analysis Insights",
    [
        f"Analyzed {report['stint_rows']:,} stint records and {report['lap_stint_rows']:,} lap-stint joins.",
        f"Most used compound: {report['most_used_compound']}.",
        f"Top stint extension profile: {report['top_stint_extension_driver']}.",
    ],
    [],
    [
        "Use compound, stint_lap_index, estimated_tyre_age, and stint_count as Gold tyre strategy features.",
        "Model degradation by compound separately; tyre behavior is not homogeneous across compounds.",
        "Treat undercut_proxy as exploratory until pit and position timing are modeled directly.",
    ],
)
(CHECKPOINTS / "silver_stint_analysis_completed.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)"""),
    ], NB_DIR / "11_stint_analysis.ipynb")


if __name__ == "__main__":
    build()

"""Build Bronze validation notebooks for the new ETL plan."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "eda" / "bronze" / "notebooks"


SETUP = """from pathlib import Path
import sys
from datetime import datetime
import json

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path.cwd()
while not (ROOT / "configs" / "pipeline_config.yaml").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

SHARED = ROOT / "eda" / "shared" / "scripts"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from config import (
    RAW_DATA_PATH,
    EXPECTED_ENDPOINTS,
    TELEMETRY_ENDPOINTS,
    CRITICAL_COLUMNS,
    PRIMARY_KEYS,
    FOREIGN_KEYS,
    RANGE_RULES,
    THRESHOLDS,
    TECHNICAL_KEY_COLUMNS,
    DOMAIN_REVIEW_COLUMNS,
    STRUCTURAL_OPTIONAL_COLUMNS,
    ALLOWED_NULL_SCENARIOS,
    VALIDATION_SEVERITY,
    RANGE_SEVERITY_OVERRIDES,
)
from file_utils import build_file_inventory, endpoint_files, endpoint_files, iter_csv_endpoint
from validation_utils import endpoint_columns, null_profile, duplicate_count, range_violations

NOTEBOOK_NAME = "__NOTEBOOK_NAME__"
OUTPUT_TABLES = ROOT / "eda" / "bronze" / "outputs" / "tables" / NOTEBOOK_NAME
OUTPUT_CHARTS = ROOT / "eda" / "bronze" / "outputs" / "charts" / NOTEBOOK_NAME
OUTPUT_REPORTS = ROOT / "eda" / "bronze" / "outputs" / "reports" / NOTEBOOK_NAME
INSIGHTS = ROOT / "eda" / "bronze" / "insights"
CHECKPOINTS = ROOT / "eda" / "bronze" / "checkpoints"
for path in [OUTPUT_TABLES, OUTPUT_CHARTS, OUTPUT_REPORTS, INSIGHTS, CHECKPOINTS]:
    path.mkdir(parents=True, exist_ok=True)

def write_report(name: str, payload: dict) -> None:
    (OUTPUT_REPORTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def write_insight(filename: str, title: str, summary: str, observations: list[str], issues: list[str], recommendations: list[str], next_steps: list[str]) -> None:
    content = f"# {title}\\n\\n"
    content += f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
    content += f"## Summary\\n\\n{summary}\\n\\n"
    content += "## Key Observations\\n\\n" + "\\n".join(f"- {item}" for item in observations) + "\\n\\n"
    content += "## Issues\\n\\n" + ("\\n".join(f"- {item}" for item in issues) if issues else "- None") + "\\n\\n"
    content += "## Recommendations\\n\\n" + "\\n".join(f"- {item}" for item in recommendations) + "\\n\\n"
    content += "## Next Steps\\n\\n" + "\\n".join(f"- {item}" for item in next_steps) + "\\n"
    (INSIGHTS / filename).write_text(content, encoding="utf-8")

def p0_status_from_severity(df: pd.DataFrame) -> str:
    if df.empty or "severity" not in df.columns:
        return "PASS"
    blockers = df[df["severity"].eq("BLOCKER")]
    return "FAIL" if not blockers.empty else "PASS"

print("=" * 72)
print(f"BRONZE VALIDATION - {NOTEBOOK_NAME}")
print(f"Start time: {datetime.now()}")
print(f"Raw data path: {RAW_DATA_PATH}")
print("=" * 72)
"""


def nb(cells: list, path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    nbformat.write(notebook, path)


def md(text: str):
    return nbformat.v4.new_markdown_cell(text)


def code(text: str):
    return nbformat.v4.new_code_cell(text)


def build() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)

    nb([
        md("# 01 File Integrity\n\nP0 validation for endpoint presence, file size, zero-row files, and corruption."),
        code(SETUP.replace("__NOTEBOOK_NAME__", "01_file_integrity")),
        code("""inventory = build_file_inventory(RAW_DATA_PATH, EXPECTED_ENDPOINTS)
inventory["endpoint_group"] = inventory["endpoint"].map(lambda endpoint: "Telemetry" if endpoint in TELEMETRY_ENDPOINTS else "Operational")
inventory["severity"] = inventory.apply(
    lambda row: "BLOCKER" if row["error"] == "endpoint_missing" or row["corrupt"] or row["zero_rows"] else "INFO",
    axis=1,
)
inventory["status"] = inventory["severity"].map(lambda severity: "FAIL" if severity == "BLOCKER" else "PASS")
inventory["decision_reason"] = inventory.apply(
    lambda row: row["error"] if row["error"] else ("zero-row file" if row["zero_rows"] else "file readable"),
    axis=1,
)
inventory.to_csv(OUTPUT_TABLES / "file_integrity.csv", index=False)

summary = (
    inventory.groupby(["endpoint_group", "endpoint"], dropna=False)
    .agg(files=("relative_path", lambda s: int((s != "").sum())), rows=("row_count", "sum"), blockers=("severity", lambda s: int((s == "BLOCKER").sum())))
    .reset_index()
)
summary["status"] = summary["blockers"].map(lambda blockers: "FAIL" if blockers else "PASS")
summary.to_csv(OUTPUT_TABLES / "file_integrity_summary.csv", index=False)
display(summary)"""),
        code("""fig = px.bar(
    summary,
    x="endpoint",
    y="files",
    color="endpoint_group",
    title="Bronze File Integrity: File Coverage by Endpoint Group",
    labels={"files": "Files", "endpoint": "Endpoint"},
)
fig.update_xaxes(tickangle=35)
fig.write_html(OUTPUT_CHARTS / "row_coverage.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUTPUT_CHARTS / "row_coverage.png")
except Exception:
    pass
fig.show()"""),
        code("""failed = inventory[inventory["status"] == "FAIL"]
report = {
    "notebook": NOTEBOOK_NAME,
    "timestamp": datetime.now().isoformat(),
    "p0_status": p0_status_from_severity(inventory),
    "files_scanned": int(len(inventory)),
    "failed_records": failed.to_dict("records"),
}
write_report("file_integrity", report)
write_insight(
    "01_file_integrity_insights.md",
    "File Integrity Insights",
    f"Scanned {len(inventory)} raw files across {inventory['endpoint'].nunique()} endpoints.",
    [
        f"Corrupt files: {int(inventory['corrupt'].sum())}",
        f"Zero-row files: {int(inventory['zero_rows'].sum())}",
        f"Telemetry rows: {int(inventory[inventory['endpoint_group'].eq('Telemetry')]['row_count'].sum()):,}",
    ],
    [f"{row.endpoint}: {row.error or 'failed integrity'}" for row in failed.itertuples()],
    ["Keep telemetry validation streaming-only.", "Proceed to schema validation if P0 status is PASS."],
    ["Run 02_schema_validation.ipynb"],
)
print(report)"""),
    ], NB_DIR / "01_file_integrity.ipynb")

    nb([
        md("# 02 Schema Validation\n\nP0 checks for required columns and P1 checks for schema drift."),
        code(SETUP.replace("__NOTEBOOK_NAME__", "02_schema_validation")),
        code("""inventory_path = ROOT / "eda" / "bronze" / "outputs" / "tables" / "01_file_integrity" / "file_integrity.csv"
if inventory_path.exists():
    inventory = pd.read_csv(inventory_path)
else:
    inventory = build_file_inventory(RAW_DATA_PATH, EXPECTED_ENDPOINTS)

records = []
for endpoint in EXPECTED_ENDPOINTS:
    endpoint_rows = inventory[inventory["endpoint"].eq(endpoint)]
    observed = []
    for value in endpoint_rows["columns"].dropna().astype(str):
        for column in value.split("|"):
            if column and column not in observed:
                observed.append(column)
    required = CRITICAL_COLUMNS.get(endpoint, [])
    missing = [column for column in required if column not in observed]
    extra = [column for column in observed if column not in required]
    records.append({
        "endpoint": endpoint,
        "required_columns": "|".join(required),
        "observed_columns": "|".join(observed),
        "missing_required_count": len(missing),
        "missing_required_columns": "|".join(missing),
        "extra_columns_count": len(extra),
        "status": "FAIL" if missing else "PASS",
    })
schema_df = pd.DataFrame(records)
schema_df.to_csv(OUTPUT_TABLES / "schema_validation.csv", index=False)
display(schema_df)"""),
        code("""plot_df = schema_df[["endpoint", "missing_required_count", "extra_columns_count"]].melt(id_vars="endpoint", var_name="metric", value_name="count")
fig = px.bar(plot_df, x="endpoint", y="count", color="metric", barmode="group", title="Schema Validation: Missing Required and Extra Columns")
fig.update_xaxes(tickangle=35)
fig.write_html(OUTPUT_CHARTS / "schema_validation.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUTPUT_CHARTS / "schema_validation.png")
except Exception:
    pass
fig.show()"""),
        code("""failed = schema_df[schema_df["status"] == "FAIL"]
report = {"notebook": NOTEBOOK_NAME, "timestamp": datetime.now().isoformat(), "p0_status": "PASS" if failed.empty else "FAIL", "results": schema_df.to_dict("records")}
write_report("schema_validation", report)
write_insight(
    "02_schema_insights.md",
    "Schema Validation Insights",
    f"Validated required schemas for {len(schema_df)} endpoints.",
    [f"Endpoints with missing required columns: {len(failed)}"],
    [f"{row.endpoint}: missing {row.missing_required_columns}" for row in failed.itertuples()],
    ["Update schema contracts or adjust crawler/cleaning logic for missing required columns."],
    ["Run 03_pk_fk_checks.ipynb"],
)
print(report["p0_status"])"""),
    ], NB_DIR / "02_schema_validation.ipynb")

    nb([
        md("# 03 Primary-Key and Foreign-Key Checks\n\nChecks natural keys and basic session-level referential integrity. Telemetry duplicate checks are skipped unless event-level keys exist."),
        code(SETUP.replace("__NOTEBOOK_NAME__", "03_pk_fk_checks")),
        code("""records = []
chunk_size = int(THRESHOLDS.get("validation_chunk_size", 200000))
for endpoint, key in PRIMARY_KEYS.items():
    if endpoint in TELEMETRY_ENDPOINTS:
        records.append({"check_type": "pk", "endpoint": endpoint, "key": "|".join(key), "rows_checked": 0, "violations": 0, "status": "SKIP"})
        continue
    rows, duplicates = duplicate_count(RAW_DATA_PATH, endpoint, key, chunk_size)
    records.append({"check_type": "pk", "endpoint": endpoint, "key": "|".join(key), "rows_checked": rows, "violations": duplicates, "status": "PASS" if duplicates == 0 else "FAIL"})

parent_cache = {}
for fk in FOREIGN_KEYS:
    parent = fk["parent"]
    parent_key = fk["parent_key"]
    if parent not in parent_cache:
        values = set()
        for chunk in iter_csv_endpoint(RAW_DATA_PATH, parent, columns=[parent_key], chunksize=chunk_size):
            if parent_key in chunk.columns:
                values.update(chunk[parent_key].dropna().astype(str).tolist())
        parent_cache[parent] = values
    child_values = set()
    for chunk in iter_csv_endpoint(RAW_DATA_PATH, fk["child"], columns=[fk["child_key"]], chunksize=chunk_size):
        if fk["child_key"] in chunk.columns:
            child_values.update(chunk[fk["child_key"]].dropna().astype(str).tolist())
    orphans = child_values - parent_cache[parent]
    records.append({"check_type": "fk", "endpoint": fk["child"], "key": f"{fk['child_key']}->{parent}.{parent_key}", "rows_checked": len(child_values), "violations": len(orphans), "status": "PASS" if not orphans else "FAIL"})

pkfk_df = pd.DataFrame(records)
pkfk_df.to_csv(OUTPUT_TABLES / "pk_fk_validation.csv", index=False)
display(pkfk_df)"""),
        code("""fig = px.bar(pkfk_df, x="endpoint", y="violations", color="check_type", barmode="group", title="PK/FK Validation Violations")
fig.update_xaxes(tickangle=35)
fig.write_html(OUTPUT_CHARTS / "pk_fk_violations.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUTPUT_CHARTS / "pk_fk_violations.png")
except Exception:
    pass
fig.show()"""),
        code("""failed = pkfk_df[pkfk_df["status"] == "FAIL"]
report = {"notebook": NOTEBOOK_NAME, "timestamp": datetime.now().isoformat(), "p0_status": "PASS" if failed.empty else "FAIL", "results": pkfk_df.to_dict("records")}
write_report("pk_fk_validation", report)
write_insight(
    "03_pk_fk_insights.md",
    "PK/FK Validation Insights",
    f"Executed {len(pkfk_df)} key-integrity checks.",
    [f"Failed checks: {len(failed)}"],
    [f"{row.check_type.upper()} {row.endpoint} {row.key}: {row.violations} violations" for row in failed.itertuples()],
    ["Investigate duplicate keys and orphan session references before Silver cleaning."],
    ["Run 04_null_analysis.ipynb"],
)
print(report["p0_status"])"""),
    ], NB_DIR / "03_pk_fk_checks.ipynb")

    nb([
        md("# 04 Null Analysis\n\nComputes null percentages using chunking/row groups and checks critical columns."),
        code(SETUP.replace("__NOTEBOOK_NAME__", "04_null_analysis")),
        code("""inventory_path = ROOT / "eda" / "bronze" / "outputs" / "tables" / "01_file_integrity" / "file_integrity.csv"
if inventory_path.exists():
    inventory = pd.read_csv(inventory_path)
else:
    inventory = build_file_inventory(RAW_DATA_PATH, EXPECTED_ENDPOINTS)

endpoint_column_cache = {}
endpoint_row_cache = {}
for endpoint in EXPECTED_ENDPOINTS:
    observed = []
    endpoint_rows = inventory[inventory["endpoint"].eq(endpoint)]
    endpoint_row_cache[endpoint] = int(endpoint_rows["row_count"].sum()) if "row_count" in endpoint_rows else 0
    for value in endpoint_rows["columns"].dropna().astype(str):
        for column in value.split("|"):
            if column and column not in observed:
                observed.append(column)
    endpoint_column_cache[endpoint] = observed

def classify_null(endpoint: str, column: str, null_count: int) -> tuple[str, str]:
    if null_count == 0:
        return "PASS", "NO_NULLS"
    if f"{endpoint}.{column}" in ALLOWED_NULL_SCENARIOS:
        return "PASS", "ALLOWED_STRUCTURAL_NULL"
    if column in TECHNICAL_KEY_COLUMNS:
        return "FAIL", "TECHNICAL_KEY_NULL"
    if column in DOMAIN_REVIEW_COLUMNS:
        return "REVIEW", "DOMAIN_SEMANTIC_NULL"
    if column in STRUCTURAL_OPTIONAL_COLUMNS:
        return "PASS", "STRUCTURAL_OR_OPTIONAL_NULL"
    if column in CRITICAL_COLUMNS.get(endpoint, []):
        return "REVIEW", "CONFIG_CRITICAL_REVIEW"
    return "PASS", "OPTIONAL_SOURCE_NULL"

profiles = []
chunk_size = int(THRESHOLDS.get("validation_chunk_size", 200000))
for endpoint in EXPECTED_ENDPOINTS:
    expected_columns = endpoint_column_cache.get(endpoint, [])
    is_telemetry = endpoint in TELEMETRY_ENDPOINTS
    profile = null_profile(
        RAW_DATA_PATH,
        endpoint,
        is_telemetry=is_telemetry,
        chunksize=chunk_size,
        expected_columns=expected_columns,
    )
    if profile.empty and expected_columns:
        profile = pd.DataFrame([
            {"endpoint": endpoint, "column": column, "rows": endpoint_row_cache.get(endpoint, 0), "null_count": 0, "null_pct": 0.0}
            for column in expected_columns
        ])
    if profile.empty:
        continue
    profile["critical"] = profile["column"].map(lambda column: column in CRITICAL_COLUMNS.get(endpoint, []))
    classified = profile.apply(lambda row: classify_null(row["endpoint"], row["column"], int(row["null_count"])), axis=1)
    profile["status"] = [item[0] for item in classified]
    profile["null_class"] = [item[1] for item in classified]
    profiles.append(profile)

null_df = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
null_df = null_df.sort_values(["status", "null_pct", "endpoint", "column"], ascending=[True, False, True, True])
null_df.to_csv(OUTPUT_TABLES / "null_percentage.csv", index=False)
display(null_df.sort_values("null_pct", ascending=False).head(80))"""),
        code("""plot_df = null_df[null_df["null_count"] > 0].sort_values("null_pct", ascending=False).head(50)
fig = px.bar(plot_df, x="null_pct", y="endpoint", color="column", orientation="h", title="Top Null Percentages by Endpoint Column")
fig.write_html(OUTPUT_CHARTS / "null_percentage.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUTPUT_CHARTS / "null_percentage.png")
except Exception:
    pass
fig.show()"""),
        code("""failed = null_df[null_df["status"] == "FAIL"] if not null_df.empty else pd.DataFrame()
review = null_df[null_df["status"] == "REVIEW"] if not null_df.empty else pd.DataFrame()
report = {"notebook": NOTEBOOK_NAME, "timestamp": datetime.now().isoformat(), "p0_status": "PASS" if failed.empty else "FAIL", "review_count": int(len(review)), "results": null_df.to_dict("records") if not null_df.empty else []}
write_report("null_analysis", report)
write_insight(
    "04_null_insights.md",
    "Null Analysis Insights",
    f"Computed null profile for {null_df['endpoint'].nunique() if not null_df.empty else 0} endpoints and {len(null_df)} endpoint-columns.",
    [f"Columns with nulls: {int((null_df['null_count'] > 0).sum()) if not null_df.empty else 0}", f"Review-level semantic nulls: {len(review)}"],
    [f"{row.endpoint}.{row.column}: {row.null_count} nulls classified as {row.null_class}" for row in failed.itertuples()],
    ["Treat technical key nulls as P0 only.", "Document domain semantic nulls as REVIEW for Silver strategy.", "Keep structural optional nulls out of P0 gating."],
    ["Run 05_range_validation.ipynb"],
)
print(report["p0_status"])"""),
    ], NB_DIR / "04_null_analysis.ipynb")

    nb([
        md("# 05 Range Validation\n\nChecks numeric domain ranges using streaming for telemetry."),
        code(SETUP.replace("__NOTEBOOK_NAME__", "05_range_validation")),
        code("""inventory_path = ROOT / "eda" / "bronze" / "outputs" / "tables" / "01_file_integrity" / "file_integrity.csv"
if inventory_path.exists():
    inventory = pd.read_csv(inventory_path)
else:
    inventory = build_file_inventory(RAW_DATA_PATH, EXPECTED_ENDPOINTS)

endpoint_column_cache = {}
for endpoint in EXPECTED_ENDPOINTS:
    observed = []
    endpoint_rows = inventory[inventory["endpoint"].eq(endpoint)]
    for value in endpoint_rows["columns"].dropna().astype(str):
        for column in value.split("|"):
            if column and column not in observed:
                observed.append(column)
    endpoint_column_cache[endpoint] = observed

records = []
chunk_size = int(THRESHOLDS.get("validation_chunk_size", 200000))
for endpoint in EXPECTED_ENDPOINTS:
    cols = endpoint_column_cache.get(endpoint, [])
    rules = {col: rule for col, rule in RANGE_RULES.items() if col in cols}
    if not rules:
        continue
    result = range_violations(RAW_DATA_PATH, endpoint, rules, endpoint in TELEMETRY_ENDPOINTS, chunksize=chunk_size)
    if not result.empty:
        result["endpoint_column"] = result["endpoint"] + "." + result["column"]
        result["severity"] = result["endpoint_column"].map(lambda key: RANGE_SEVERITY_OVERRIDES.get(key, "WARNING"))
        result["status"] = result.apply(lambda row: "PASS" if row["violations"] == 0 else ("FAIL" if row["severity"] == "BLOCKER" else "WARN"), axis=1)
        records.append(result)
range_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
range_df.to_csv(OUTPUT_TABLES / "range_violations.csv", index=False)
display(range_df.sort_values("violations", ascending=False).head(50))"""),
        code("""fig = px.bar(range_df, x="endpoint", y="violations", color="column", barmode="group", title="Range Validation Violations")
fig.update_xaxes(tickangle=35)
fig.write_html(OUTPUT_CHARTS / "range_violations.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUTPUT_CHARTS / "range_violations.png")
except Exception:
    pass
fig.show()"""),
        code("""failed = range_df[(range_df["severity"] == "BLOCKER") & (range_df["violations"] > 0)] if not range_df.empty else pd.DataFrame()
warned = range_df[(range_df["violations"] > 0) & (range_df["severity"] != "BLOCKER")] if not range_df.empty else pd.DataFrame()
report = {"notebook": NOTEBOOK_NAME, "timestamp": datetime.now().isoformat(), "p0_status": "PASS" if failed.empty else "FAIL", "warning_count": int(len(warned)), "results": range_df.to_dict("records") if not range_df.empty else []}
write_report("range_validation", report)
write_insight(
    "05_range_insights.md",
    "Range Validation Insights",
    f"Validated numeric range rules across {range_df['endpoint'].nunique() if not range_df.empty else 0} endpoints.",
    [f"Columns checked: {range_df['column'].nunique() if not range_df.empty else 0}", f"Non-blocking range warnings: {len(warned)}"],
    [f"{row.endpoint}.{row.column}: {row.violations} BLOCKER violations" for row in failed.itertuples()],
    ["Use min_violation, max_violation, and sample_values columns to decide Silver mitigation.", "Treat weather pressure and lap-duration violations as contextual warnings unless confirmed impossible."],
    ["Run 06_temporal_checks.ipynb"],
)
print(report["p0_status"])"""),
    ], NB_DIR / "05_range_validation.ipynb")

    nb([
        md("# 06 Temporal Checks\n\nChecks session date logic and lap-number continuity."),
        code(SETUP.replace("__NOTEBOOK_NAME__", "06_temporal_checks")),
        code("""records = []
sessions = []
for chunk in iter_csv_endpoint(RAW_DATA_PATH, "sessions", chunksize=200000):
    sessions.append(chunk)
sessions_df = pd.concat(sessions, ignore_index=True) if sessions else pd.DataFrame()
if {"session_key", "date_start", "date_end"}.issubset(sessions_df.columns):
    starts = pd.to_datetime(sessions_df["date_start"], errors="coerce", utc=True)
    ends = pd.to_datetime(sessions_df["date_end"], errors="coerce", utc=True)
    invalid = int((starts.notna() & ends.notna() & (starts >= ends)).sum())
    records.append({"check": "session_date_order", "scope": "sessions", "violations": invalid, "severity": "BLOCKER", "status": "PASS" if invalid == 0 else "FAIL"})

lap_frames = []
for chunk in iter_csv_endpoint(RAW_DATA_PATH, "laps", columns=["session_key", "driver_number", "lap_number"], chunksize=200000):
    lap_frames.append(chunk[["session_key", "driver_number", "lap_number"]])
laps_df = pd.concat(lap_frames, ignore_index=True) if lap_frames else pd.DataFrame()
gap_count = 0
affected = 0
if not laps_df.empty:
    laps_df["lap_number"] = pd.to_numeric(laps_df["lap_number"], errors="coerce")
    valid_laps = laps_df.dropna(subset=["session_key", "driver_number", "lap_number"]).copy()
    valid_laps["lap_number"] = valid_laps["lap_number"].astype(int)
    for _, group in valid_laps.groupby(["session_key", "driver_number"], sort=False):
        observed = np.sort(group["lap_number"].unique())
        if observed.size == 0:
            continue
        gaps = np.diff(observed) - 1
        missing_count = int(gaps[gaps > 0].sum())
        if missing_count:
            affected += 1
            gap_count += missing_count
records.append({"check": "lap_number_continuity", "scope": "laps", "violations": gap_count, "affected_groups": affected, "severity": "WARNING", "status": "PASS" if gap_count == 0 else "WARN"})

temporal_df = pd.DataFrame(records)
temporal_df.to_csv(OUTPUT_TABLES / "temporal_issues.csv", index=False)
display(temporal_df)"""),
        code("""fig = px.bar(temporal_df, x="check", y="violations", color="status", title="Temporal Validation Issues")
fig.write_html(OUTPUT_CHARTS / "temporal_issues.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUTPUT_CHARTS / "temporal_issues.png")
except Exception:
    pass
fig.show()"""),
        code("""failed = temporal_df[(temporal_df["severity"] == "BLOCKER") & (temporal_df["violations"] > 0)]
warned = temporal_df[(temporal_df["severity"] != "BLOCKER") & (temporal_df["violations"] > 0)]
report = {"notebook": NOTEBOOK_NAME, "timestamp": datetime.now().isoformat(), "p0_status": "PASS" if failed.empty else "FAIL", "warning_count": int(len(warned)), "results": temporal_df.to_dict("records")}
write_report("temporal_validation", report)
write_insight(
    "06_temporal_insights.md",
    "Temporal Validation Insights",
    f"Executed {len(temporal_df)} temporal checks.",
    [f"Lap continuity gaps: {gap_count}", f"Affected driver-session groups: {affected}", "Lap gaps are non-blocking until Silver DNF classification is available."],
    [f"{row.check}: {row.violations} violations" for row in failed.itertuples()],
    ["Classify lap gaps as DNF/pit/source gaps before feature engineering."],
    ["Run 00_final_summary.ipynb"],
)
print(report["p0_status"])"""),
    ], NB_DIR / "06_temporal_checks.ipynb")

    nb([
        md("# 00 Final Bronze Summary\n\nAggregates Bronze reports and creates the Bronze checkpoint when the gate can proceed. Outcomes are STOP for true blockers, PROCEED_WITH_WARNINGS for Silver-resolvable issues, and PROCEED when all checks are clean."),
        code(SETUP.replace("__NOTEBOOK_NAME__", "00_final_summary")),
        code("""report_files = [
    path for path in sorted((ROOT / "eda" / "bronze" / "outputs" / "reports").glob("*/*.json"))
    if path.parent.name != "00_final_summary"
]
rows = []
for path in report_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "report": path.stem,
        "notebook": payload.get("notebook"),
        "p0_status": payload.get("p0_status", "UNKNOWN"),
        "warning_count": int(payload.get("warning_count", 0) or 0),
        "review_count": int(payload.get("review_count", 0) or 0),
        "path": str(path.relative_to(ROOT)),
    })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUTPUT_TABLES / "bronze_validation_summary.csv", index=False)
display(summary_df)"""),
        code("""fig = px.bar(summary_df.groupby("p0_status", as_index=False).size(), x="p0_status", y="size", color="p0_status", title="Bronze P0 Gate Summary")
fig.write_html(OUTPUT_CHARTS / "bronze_p0_summary.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUTPUT_CHARTS / "bronze_p0_summary.png")
except Exception:
    pass
fig.show()"""),
        code("""failed = summary_df[summary_df["p0_status"] != "PASS"]
warning_total = int(summary_df["warning_count"].sum()) if "warning_count" in summary_df else 0
review_total = int(summary_df["review_count"].sum()) if "review_count" in summary_df else 0
status = "STOP" if not failed.empty else ("PROCEED_WITH_WARNINGS" if warning_total or review_total else "PROCEED")
checkpoint = {
    "status": status,
    "timestamp": datetime.now().isoformat(),
    "p0_checks_passed": int((summary_df["p0_status"] == "PASS").sum()),
    "p0_checks_total": int(len(summary_df)),
    "warning_count": warning_total,
    "review_count": review_total,
    "failed_reports": failed.to_dict("records"),
    "next_step": "Proceed to Silver with documented mitigations" if status == "PROCEED_WITH_WARNINGS" else ("Proceed to Silver layer" if status == "PROCEED" else "Resolve Bronze blockers"),
}
write_report("bronze_validation_complete", checkpoint)
write_insight(
    "00_final_summary.md",
    "Bronze Final Summary",
    f"Bronze gate status: {status}.",
    [f"P0 passed: {checkpoint['p0_checks_passed']}/{checkpoint['p0_checks_total']}", f"Warnings: {warning_total}", f"Review items: {review_total}"],
    [f"{row.report}: {row.p0_status}" for row in failed.itertuples()],
    ["STOP only on missing/corrupt files, missing critical schema, PK duplicates, FK orphans, or technical key nulls.", "Non-blocking warnings must be handled or explicitly documented in Silver."],
    [checkpoint["next_step"]],
)
if status in {"PROCEED", "PROCEED_WITH_WARNINGS"}:
    (CHECKPOINTS / "bronze_completed.txt").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
else:
    checkpoint_path = CHECKPOINTS / "bronze_completed.txt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
print(checkpoint)"""),
    ], NB_DIR / "00_final_summary.ipynb")


if __name__ == "__main__":
    build()

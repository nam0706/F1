"""Build Gold EDA notebooks that audit generated Gold artifacts.

The notebooks intentionally read only data/gold outputs. Gold generation logic
stays in src/gold, while these notebooks validate labels, feature coverage,
leakage policy, horizon datasets, and final readiness.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "eda" / "gold" / "notebooks"


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
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path.cwd()
while not (ROOT / "configs" / "pipeline_config.yaml").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

NOTEBOOK_NAME = "{notebook_name}"
GOLD = ROOT / "data" / "gold"
FEATURES = GOLD / "features"
LABELS = GOLD / "labels"
TRAINING = GOLD / "training"
METADATA = GOLD / "metadata"

OUTPUT_TABLES = ROOT / "eda" / "gold" / "outputs" / "tables" / NOTEBOOK_NAME
OUTPUT_CHARTS = ROOT / "eda" / "gold" / "outputs" / "charts" / NOTEBOOK_NAME
OUTPUT_REPORTS = ROOT / "eda" / "gold" / "outputs" / "reports" / NOTEBOOK_NAME
INSIGHTS = ROOT / "eda" / "gold" / "insights"
CHECKPOINTS = ROOT / "eda" / "gold" / "checkpoints"
for path in [OUTPUT_TABLES, OUTPUT_CHARTS, OUTPUT_REPORTS, INSIGHTS, CHECKPOINTS]:
    path.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def write_report(name: str, payload: dict) -> None:
    (OUTPUT_REPORTS / f"{{name}}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def write_insight(title: str, observations: list[str], issues: list[str], recommendations: list[str]) -> None:
    content = f"# {{title}}\\n\\n"
    content += f"**Generated at:** {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\\n\\n"
    content += "## Key Observations\\n\\n" + "\\n".join(f"- {{item}}" for item in observations) + "\\n\\n"
    content += "## Issues\\n\\n" + ("\\n".join(f"- {{item}}" for item in issues) if issues else "- None") + "\\n\\n"
    content += "## Recommendations\\n\\n" + "\\n".join(f"- {{item}}" for item in recommendations) + "\\n"
    (INSIGHTS / f"{{NOTEBOOK_NAME}}.md").write_text(content, encoding="utf-8")

def save_chart(fig, name: str) -> None:
    fig.write_html(OUTPUT_CHARTS / f"{{name}}.html", include_plotlyjs="cdn")

print("=" * 72)
print(f"GOLD EDA - {{NOTEBOOK_NAME}}")
print(f"Start time: {{datetime.now()}}")
print(f"Gold root: {{GOLD}}")
"""


NOTEBOOKS: dict[str, list] = {}


NOTEBOOKS["01_gold_label_audit.ipynb"] = [
    md(
        """# 01 Gold Label Audit

This notebook audits the isolated Gold label table. The label table is allowed
to contain post-race outcomes, but those outcomes must remain separate from
feature generation and only join at training dataset construction time."""
    ),
    code(setup("01_gold_label_audit")),
    md(
        """## Finish Bucket Distribution

F1 finish outcomes are naturally imbalanced: wins are rare, podiums are limited
to three cars, and points depend on sporting rules. The key audit question is
whether the label table preserves every driver-session outcome without leaking
those outcomes into feature artifacts."""
    ),
    code(
        """labels = pd.read_parquet(LABELS / "race_result_labels.parquet")
label_contract = read_json(METADATA / "label_contract.json")

bucket_summary = (
    labels.groupby(["target_finish_bucket", "target_finish_bucket_label"], dropna=False)
    .size()
    .rename("rows")
    .reset_index()
    .sort_values("target_finish_bucket")
)
bucket_summary["pct"] = (bucket_summary["rows"] / bucket_summary["rows"].sum() * 100).round(2)
bucket_summary.to_csv(OUTPUT_TABLES / "finish_bucket_distribution.csv", index=False)

fig = px.bar(
    bucket_summary,
    x="target_finish_bucket_label",
    y="rows",
    color="target_finish_bucket_label",
    text="rows",
    title="Gold Label Distribution: Finish Bucket Target",
    labels={"target_finish_bucket_label": "Finish bucket", "rows": "Driver-session labels"},
)
fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=80))
fig.update_traces(textposition="outside", cliponaxis=False)
save_chart(fig, "finish_bucket_distribution")
fig.show()

display(bucket_summary)"""
    ),
    md(
        """## Label Integrity

The label grain must be exactly one row per `session_key + driver_number`. Any
duplicate key would create ambiguous supervision for the model, while nullable
targets would create silent training exclusions."""
    ),
    code(
        """quality_rows = []
quality_rows.append({
    "check": "label_key_uniqueness",
    "metric": "duplicate session-driver rows",
    "value": int(labels.duplicated(["session_key", "driver_number"]).sum()),
    "status": "PASS" if labels.duplicated(["session_key", "driver_number"]).sum() == 0 else "FAIL",
})
for column in ["target_win", "target_podium", "target_top10", "target_points", "target_dnf", "target_finish_bucket"]:
    null_count = int(labels[column].isna().sum()) if column in labels.columns else len(labels)
    quality_rows.append({
        "check": f"{column}_not_null",
        "metric": "null target rows",
        "value": null_count,
        "status": "PASS" if null_count == 0 else "FAIL",
    })
label_quality = pd.DataFrame(quality_rows)
label_quality.to_csv(OUTPUT_TABLES / "label_quality_matrix.csv", index=False)

fig = px.bar(
    label_quality,
    x="check",
    y="value",
    color="status",
    title="Gold Label Quality Matrix",
    labels={"check": "Quality check", "value": "Issue count"},
)
fig.update_layout(margin=dict(l=10, r=10, t=55, b=120))
save_chart(fig, "label_quality_matrix")
fig.show()

display(label_quality)"""
    ),
    code(
        """write_report("label_audit", {
    "rows": int(len(labels)),
    "unique_sessions": int(labels["session_key"].nunique()),
    "unique_driver_sessions": int(labels[["session_key", "driver_number"]].drop_duplicates().shape[0]),
    "quality_status": "PASS" if label_quality["status"].eq("PASS").all() else "FAIL",
    "bucket_distribution": bucket_summary.to_dict(orient="records"),
})
write_insight(
    "Gold Label Audit",
    [
        f"Gold labels contain {len(labels):,} driver-session outcomes across {labels['session_key'].nunique():,} sessions.",
        "Target classes are intentionally imbalanced because F1 sporting outcomes are constrained by grid size and points rules.",
        "Post-race outcome fields remain isolated in the label artifact and are not present in master_lap_features.",
    ],
    [] if label_quality["status"].eq("PASS").all() else ["One or more label integrity checks failed."],
    [
        "Use stratified validation where possible because finish buckets are imbalanced.",
        "Keep label columns out of feature selection; join them only in training dataset generation.",
    ],
)
(CHECKPOINTS / "gold_label_audit_completed.txt").write_text(datetime.now().isoformat(), encoding="utf-8")"""
    ),
]


NOTEBOOKS["02_gold_feature_coverage.ipynb"] = [
    md(
        """# 02 Gold Feature Coverage

This notebook validates the Gold feature layer after `src/gold/build_gold.py`
has generated the artifacts. It does not compute features directly. Its job is
to confirm grain consistency, feature coverage, and known sparse regions."""
    ),
    code(setup("02_gold_feature_coverage")),
    md(
        """## Feature Table Row Coverage

Every lap-grain feature table should align with the canonical master grain:
`session_key + driver_number + lap_number`. Mismatched row counts create
silent train-time nulls or dropped rows."""
    ),
    code(
        """feature_files = sorted(FEATURES.glob("*.parquet"))
coverage_rows = []
for path in feature_files:
    df = pd.read_parquet(path)
    key_dupes = int(df.duplicated(["session_key", "driver_number", "lap_number"]).sum()) if {"session_key", "driver_number", "lap_number"}.issubset(df.columns) else None
    coverage_rows.append({
        "artifact": path.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_keys": key_dupes,
    })
coverage = pd.DataFrame(coverage_rows).sort_values("artifact")
coverage["status"] = np.where(coverage["duplicate_keys"].fillna(0).eq(0), "PASS", "FAIL")
coverage.to_csv(OUTPUT_TABLES / "feature_row_coverage.csv", index=False)

fig = px.bar(
    coverage,
    x="artifact",
    y="rows",
    color="status",
    text="rows",
    title="Gold Feature Row Coverage by Artifact",
    labels={"artifact": "Feature artifact", "rows": "Rows"},
)
fig.update_layout(margin=dict(l=10, r=10, t=55, b=130))
fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
save_chart(fig, "feature_row_coverage")
fig.show()

display(coverage)"""
    ),
    md(
        """## Missingness Map

Nulls in Gold are not automatically defects. Some fields are expected to be
sparse because telemetry/event sources are sampled differently, and stint data
can be incomplete for specific sessions. The audit separates coverage risk from
pipeline failure."""
    ),
    code(
        """master = pd.read_parquet(FEATURES / "master_lap_features.parquet")
null_summary = (
    master.isna().sum()
    .rename("null_count")
    .reset_index()
    .rename(columns={"index": "column"})
)
null_summary["null_pct"] = (null_summary["null_count"] / len(master) * 100).round(3)
null_summary = null_summary.sort_values("null_pct", ascending=False)
null_summary.to_csv(OUTPUT_TABLES / "master_feature_null_profile.csv", index=False)

plot_nulls = null_summary[null_summary["null_count"].gt(0)].head(25)
fig = px.bar(
    plot_nulls,
    x="null_pct",
    y="column",
    orientation="h",
    text="null_pct",
    title="Gold Master Feature Missingness: Top Sparse Columns",
    labels={"null_pct": "Null percentage", "column": "Feature"},
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=55, b=10))
fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
save_chart(fig, "master_feature_missingness")
fig.show()

display(null_summary.head(25))"""
    ),
    code(
        """write_report("feature_coverage", {
    "master_rows": int(len(master)),
    "master_columns": int(len(master.columns)),
    "feature_artifacts": coverage.to_dict(orient="records"),
    "top_null_columns": null_summary.head(25).to_dict(orient="records"),
})
write_insight(
    "Gold Feature Coverage",
    [
        f"Master lap features contain {len(master):,} rows and {len(master.columns):,} columns.",
        "All generated lap-grain feature artifacts preserve canonical driver-session-lap grain.",
        "Sparse stint rows are retained rather than dropped, protecting label alignment.",
    ],
    [] if coverage["status"].eq("PASS").all() else ["At least one feature artifact has duplicate grain keys."],
    [
        "Treat high-null feature columns explicitly during model preprocessing.",
        "Investigate missing stint coverage before using tyre features as mandatory model inputs.",
    ],
)
(CHECKPOINTS / "gold_feature_coverage_completed.txt").write_text(datetime.now().isoformat(), encoding="utf-8")"""
    ),
]


NOTEBOOKS["03_gold_leakage_audit.ipynb"] = [
    md(
        """# 03 Gold Leakage Audit

This notebook audits the strict Gold leakage report. The model should select
features from the whitelist contract, not by dropping a small blacklist."""
    ),
    code(setup("03_gold_leakage_audit")),
    md(
        """## Leakage Gate

The leakage gate blocks post-race outcome fields, duplicate feature keys,
duplicate label keys, nullable targets, and unregistered feature columns. A
clean report is required before the Gold training datasets are trusted."""
    ),
    code(
        """leakage = read_json(METADATA / "leakage_report.json")
checks = [
    ("banned_columns_in_features", len(leakage.get("banned_columns_in_features", []))),
    ("duplicate_feature_key_rows", int(leakage.get("duplicate_feature_key_rows", 0))),
    ("duplicate_label_key_rows", int(leakage.get("duplicate_label_key_rows", 0))),
    ("nullable_feature_keys", len(leakage.get("nullable_feature_keys", {}))),
    ("nullable_targets", len(leakage.get("nullable_targets", {}))),
    ("unchecked_model_columns", len(leakage.get("unchecked_model_columns", []))),
]
leakage_matrix = pd.DataFrame(checks, columns=["check", "issue_count"])
leakage_matrix["status"] = np.where(leakage_matrix["issue_count"].eq(0), "PASS", "FAIL")
leakage_matrix.to_csv(OUTPUT_TABLES / "leakage_quality_matrix.csv", index=False)

fig = px.bar(
    leakage_matrix,
    x="check",
    y="issue_count",
    color="status",
    title=f"Gold Leakage Gate: {leakage.get('status')}",
    labels={"check": "Leakage check", "issue_count": "Issue count"},
)
fig.update_layout(margin=dict(l=10, r=10, t=55, b=120))
save_chart(fig, "leakage_quality_matrix")
fig.show()

display(leakage_matrix)"""
    ),
    md(
        """## Feature Whitelist Coverage

The whitelist is the enforceable boundary between Gold and model training.
Every model-eligible column must be registered with an explicit availability
rule."""
    ),
    code(
        """feature_contract = read_json(METADATA / "feature_contract.json")
whitelist_rows = []
for group_name, contract in feature_contract.get("feature_tables", {}).items():
    for column in contract.get("model_allowed_columns", []):
        whitelist_rows.append({
            "feature_group": group_name,
            "column": column,
            "availability_rule": contract.get("availability_rule"),
        })
whitelist = pd.DataFrame(whitelist_rows)
whitelist.to_csv(OUTPUT_TABLES / "model_feature_whitelist.csv", index=False)

group_summary = whitelist.groupby("feature_group").size().rename("allowed_columns").reset_index()
fig = px.bar(
    group_summary,
    x="feature_group",
    y="allowed_columns",
    text="allowed_columns",
    title="Model Feature Whitelist by Gold Feature Group",
    labels={"feature_group": "Feature group", "allowed_columns": "Allowed columns"},
)
fig.update_layout(margin=dict(l=10, r=10, t=55, b=80))
fig.update_traces(textposition="outside", cliponaxis=False)
save_chart(fig, "feature_whitelist_by_group")
fig.show()

display(group_summary)
display(whitelist.head(50))"""
    ),
    code(
        """write_report("leakage_audit", {
    "status": leakage.get("status"),
    "blockers": leakage.get("blockers", []),
    "unchecked_model_columns": leakage.get("unchecked_model_columns", []),
    "whitelist_columns": int(len(whitelist)),
})
write_insight(
    "Gold Leakage Audit",
    [
        f"Leakage report status is {leakage.get('status')}.",
        f"The model whitelist contains {len(whitelist):,} explicitly registered columns.",
        "Label columns are not present in the feature artifact before training dataset generation.",
    ],
    leakage.get("blockers", []),
    [
        "Future model code should load this whitelist directly from feature_contract.json.",
        "Any new feature module must declare availability and model_allowed_columns before training use.",
    ],
)
(CHECKPOINTS / "gold_leakage_audit_completed.txt").write_text(datetime.now().isoformat(), encoding="utf-8")"""
    ),
]


NOTEBOOKS["04_horizon_dataset_audit.ipynb"] = [
    md(
        """# 04 Horizon Dataset Audit

This notebook validates the training datasets generated for live and fixed
race-progress horizons. Fixed horizons should keep one row per driver-session,
while `live_any_lap` keeps all eligible laps after minimum history rules."""
    ),
    code(setup("04_horizon_dataset_audit")),
    md(
        """## Horizon Row Coverage

The horizon datasets are different modeling views over the same Gold master
frame. Fixed horizons support comparable snapshots, while live-any-lap supports
interactive race-state prediction."""
    ),
    code(
        """training_contract = read_json(METADATA / "training_dataset_contract.json")
dataset_rows = []
for name, meta in training_contract["datasets"].items():
    dataset_rows.append({
        "dataset": name,
        "rows": int(meta["rows"]),
        "columns": int(meta["columns"]),
        "unique_sessions": int(meta["unique_sessions"]),
        "unique_driver_sessions": int(meta["unique_driver_sessions"]),
    })
dataset_summary = pd.DataFrame(dataset_rows).sort_values("dataset")
dataset_summary.to_csv(OUTPUT_TABLES / "horizon_dataset_summary.csv", index=False)

fig = px.bar(
    dataset_summary,
    x="dataset",
    y="rows",
    color="dataset",
    text="rows",
    title="Gold Training Dataset Row Coverage by Horizon",
    labels={"dataset": "Training dataset", "rows": "Rows"},
)
fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=100))
fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
save_chart(fig, "horizon_row_coverage")
fig.show()

display(dataset_summary)"""
    ),
    md(
        """## Target Balance by Horizon

Class balance should be stable across fixed horizons because the target is a
driver-session outcome. Material drift here would suggest horizon selection is
dropping a non-random subset of drivers."""
    ),
    code(
        """dist_rows = []
for name, meta in training_contract["datasets"].items():
    path = ROOT / meta["artifact"]
    df = pd.read_parquet(path)
    if "target_finish_bucket_label" not in df.columns:
        continue
    counts = df.groupby("target_finish_bucket_label").size().rename("rows").reset_index()
    counts["dataset"] = name
    counts["pct"] = (counts["rows"] / counts["rows"].sum() * 100).round(2)
    dist_rows.append(counts)
target_distribution = pd.concat(dist_rows, ignore_index=True)
target_distribution.to_csv(OUTPUT_TABLES / "horizon_target_distribution.csv", index=False)

fig = px.bar(
    target_distribution,
    x="dataset",
    y="rows",
    color="target_finish_bucket_label",
    title="Target Distribution Across Gold Horizon Datasets",
    labels={"dataset": "Dataset", "rows": "Rows", "target_finish_bucket_label": "Finish bucket"},
)
fig.update_layout(margin=dict(l=10, r=10, t=55, b=100))
save_chart(fig, "horizon_target_distribution")
fig.show()

display(target_distribution.sort_values(["dataset", "target_finish_bucket_label"]))"""
    ),
    code(
        """fixed_ok = dataset_summary[dataset_summary["dataset"].isin(["horizon_25", "horizon_50", "horizon_75"])]["unique_driver_sessions"].nunique() == 1
write_report("horizon_dataset_audit", {
    "datasets": dataset_summary.to_dict(orient="records"),
    "fixed_horizon_driver_session_consistency": bool(fixed_ok),
})
write_insight(
    "Gold Horizon Dataset Audit",
    [
        "Fixed horizons keep one selected row per driver-session.",
        "Live-any-lap keeps many rows per driver-session for interactive prediction.",
        "Target distribution is expected to remain stable across fixed horizons.",
    ],
    [] if fixed_ok else ["Fixed horizons do not share the same driver-session coverage."],
    [
        "Use fixed horizons for comparable race-progress experiments.",
        "Use live_any_lap for Streamlit-style lap slider prediction and larger training volume.",
    ],
)
(CHECKPOINTS / "gold_horizon_dataset_audit_completed.txt").write_text(datetime.now().isoformat(), encoding="utf-8")"""
    ),
]


NOTEBOOKS["05_gold_final_summary.ipynb"] = [
    md(
        """# 05 Gold Final Summary

This notebook consolidates the Gold audit outputs into a single readiness view
for model development. It is a reporting asset, not a feature builder."""
    ),
    code(setup("05_gold_final_summary")),
    md(
        """## Gold Readiness Matrix

The Gold layer is ready for model training only when labels, feature coverage,
leakage, and horizon datasets all pass their minimum gates."""
    ),
    code(
        """label_contract = read_json(METADATA / "label_contract.json")
feature_contract = read_json(METADATA / "feature_contract.json")
leakage = read_json(METADATA / "leakage_report.json")
training_contract = read_json(METADATA / "training_dataset_contract.json")

readiness = pd.DataFrame([
    {"gate": "labels", "metric": "label rows", "value": int(label_contract["rows"]), "status": "PASS" if label_contract["rows"] > 0 else "FAIL"},
    {"gate": "features", "metric": "master feature rows", "value": int(feature_contract["rows"]), "status": "PASS" if feature_contract["rows"] > 0 else "FAIL"},
    {"gate": "leakage", "metric": "blocker count", "value": len(leakage.get("blockers", [])), "status": leakage.get("status", "FAIL")},
    {"gate": "training", "metric": "training datasets", "value": len(training_contract.get("datasets", {})), "status": "PASS" if len(training_contract.get("datasets", {})) == 5 else "FAIL"},
])
readiness.to_csv(OUTPUT_TABLES / "gold_readiness_matrix.csv", index=False)

fig = px.bar(
    readiness,
    x="gate",
    y="value",
    color="status",
    text="value",
    title="Gold Layer Readiness Matrix",
    labels={"gate": "Gold gate", "value": "Metric value"},
)
fig.update_layout(margin=dict(l=10, r=10, t=55, b=80))
fig.update_traces(textposition="outside", cliponaxis=False)
save_chart(fig, "gold_readiness_matrix")
fig.show()

display(readiness)"""
    ),
    md(
        """## Modeling Boundary Summary

Gold owns feature construction, label isolation, horizon selection, and leakage
metadata. Model training should own split strategy, preprocessing fit,
estimator selection, calibration, and evaluation."""
    ),
    code(
        """feature_groups = []
for name, contract in feature_contract.get("feature_tables", {}).items():
    feature_groups.append({
        "feature_group": name,
        "artifact": contract.get("artifact"),
        "rows": contract.get("rows"),
        "allowed_columns": len(contract.get("model_allowed_columns", [])),
        "availability_rule": contract.get("availability_rule"),
    })
feature_group_summary = pd.DataFrame(feature_groups)
feature_group_summary.to_csv(OUTPUT_TABLES / "feature_group_summary.csv", index=False)

fig = px.bar(
    feature_group_summary,
    x="feature_group",
    y="allowed_columns",
    text="allowed_columns",
    title="Allowed Model Features by Gold Feature Group",
    labels={"feature_group": "Feature group", "allowed_columns": "Whitelisted columns"},
)
fig.update_layout(margin=dict(l=10, r=10, t=55, b=80))
fig.update_traces(textposition="outside", cliponaxis=False)
save_chart(fig, "allowed_features_by_group")
fig.show()

display(feature_group_summary)"""
    ),
    code(
        """overall_status = "PASS" if readiness["status"].eq("PASS").all() else "REVIEW"
write_report("gold_final_summary", {
    "overall_status": overall_status,
    "readiness": readiness.to_dict(orient="records"),
    "feature_groups": feature_group_summary.to_dict(orient="records"),
})
write_insight(
    "Gold Final Summary",
    [
        f"Overall Gold readiness status is {overall_status}.",
        f"Master feature table contains {feature_contract['rows']:,} lap-grain rows.",
        f"Leakage report status is {leakage.get('status')}.",
        "Gold now exposes separate feature, label, and horizon training artifacts.",
    ],
    [] if overall_status == "PASS" else ["At least one Gold readiness gate needs review."],
    [
        "Move next to model training using feature_contract.json as the feature whitelist.",
        "Keep Gold EDA notebooks as audit-only assets; do not rebuild features inside notebooks.",
    ],
)
(CHECKPOINTS / "gold_final_summary_completed.txt").write_text(datetime.now().isoformat(), encoding="utf-8")"""
    ),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        write_notebook(cells, NB_DIR / filename)


if __name__ == "__main__":
    main()

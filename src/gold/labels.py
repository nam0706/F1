from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd


FINISH_BUCKET_LABELS = {
    0: "WIN",
    1: "PODIUM_NON_WIN",
    2: "POINTS_NON_PODIUM",
    3: "CLASSIFIED_OUTSIDE_POINTS",
    4: "DNF_DNS_DSQ_UNCLASSIFIED",
}

LABEL_SOURCE_COLUMNS = [
    "position",
    "number_of_laps",
    "points",
    "dnf",
    "dns",
    "dsq",
    "duration",
    "gap_to_leader",
]

MODEL_BANNED_LABEL_SOURCE_COLUMNS = [
    "final_position",
    "classified_laps",
    "points",
    "dnf",
    "dns",
    "dsq",
    "duration",
    "gap_to_leader",
    "target_win",
    "target_podium",
    "target_top10",
    "target_points",
    "target_dnf",
    "target_finish_bucket",
]


def build_race_result_labels(session_result: pd.DataFrame) -> pd.DataFrame:
    """Build driver-session labels from Silver session_result.

    Labels are allowed to use post-race outcomes, but the output remains
    isolated from feature generation and is joined only during training dataset
    construction.
    """
    required = {"session_key", "driver_number"}
    missing = sorted(required - set(session_result.columns))
    if missing:
        raise ValueError(f"session_result is missing required columns: {missing}")

    labels = session_result.copy()
    labels["session_key"] = pd.to_numeric(labels["session_key"], errors="coerce")
    labels["driver_number"] = pd.to_numeric(labels["driver_number"], errors="coerce")
    labels = labels.dropna(subset=["session_key", "driver_number"]).copy()
    labels["session_key"] = labels["session_key"].astype("int64")
    labels["driver_number"] = labels["driver_number"].astype("int64")

    labels["final_position"] = pd.to_numeric(labels.get("position"), errors="coerce")
    labels["classified_laps"] = pd.to_numeric(labels.get("number_of_laps"), errors="coerce")
    labels["points"] = pd.to_numeric(labels.get("points"), errors="coerce").fillna(0.0)

    for column in ["dnf", "dns", "dsq"]:
        if column not in labels.columns:
            labels[column] = False
        labels[column] = labels[column].fillna(False).astype(bool)

    labels["target_win"] = labels["final_position"].eq(1).astype("int8")
    labels["target_podium"] = labels["final_position"].between(1, 3, inclusive="both").astype("int8")
    labels["target_top10"] = labels["final_position"].between(1, 10, inclusive="both").astype("int8")
    labels["target_points"] = labels["points"].gt(0).astype("int8")
    labels["target_dnf"] = (
        labels[["dnf", "dns", "dsq"]].any(axis=1) | labels["final_position"].isna()
    ).astype("int8")

    labels["target_finish_bucket"] = 3
    labels.loc[labels["target_dnf"].eq(1), "target_finish_bucket"] = 4
    labels.loc[labels["target_points"].eq(1), "target_finish_bucket"] = 2
    labels.loc[labels["target_podium"].eq(1), "target_finish_bucket"] = 1
    labels.loc[labels["target_win"].eq(1), "target_finish_bucket"] = 0
    labels["target_finish_bucket"] = labels["target_finish_bucket"].astype("int8")
    labels["target_finish_bucket_label"] = labels["target_finish_bucket"].map(FINISH_BUCKET_LABELS)
    labels["label_source"] = "session_result"

    output_columns = [
        "session_key",
        "driver_number",
        "final_position",
        "classified_laps",
        "points",
        "dnf",
        "dns",
        "dsq",
        "target_win",
        "target_podium",
        "target_top10",
        "target_points",
        "target_dnf",
        "target_finish_bucket",
        "target_finish_bucket_label",
        "label_source",
    ]
    labels = labels[output_columns].drop_duplicates(["session_key", "driver_number"], keep="first")
    labels = labels.sort_values(["session_key", "driver_number"]).reset_index(drop=True)
    return labels


def build_label_contract(labels: pd.DataFrame) -> dict:
    bucket_counts = (
        labels["target_finish_bucket"]
        .value_counts(dropna=False)
        .sort_index()
        .rename(lambda value: str(int(value)) if pd.notna(value) else "NA")
        .to_dict()
    )
    return {
        "generated_at": datetime.now().isoformat(),
        "artifact": "data/gold/labels/race_result_labels.parquet",
        "grain": ["session_key", "driver_number"],
        "rows": int(len(labels)),
        "unique_sessions": int(labels["session_key"].nunique()) if "session_key" in labels.columns else 0,
        "unique_drivers": int(labels["driver_number"].nunique()) if "driver_number" in labels.columns else 0,
        "finish_bucket_labels": FINISH_BUCKET_LABELS,
        "finish_bucket_distribution": bucket_counts,
        "label_source_columns": LABEL_SOURCE_COLUMNS,
        "model_banned_columns": MODEL_BANNED_LABEL_SOURCE_COLUMNS,
        "target_columns": [
            "target_win",
            "target_podium",
            "target_top10",
            "target_points",
            "target_dnf",
            "target_finish_bucket",
        ],
    }


def write_race_result_labels(session_result: pd.DataFrame, labels_path: Path, metadata_path: Path) -> dict:
    labels = build_race_result_labels(session_result)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_path, index=False, compression="zstd")
    contract = build_label_contract(labels)
    metadata_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
    return contract

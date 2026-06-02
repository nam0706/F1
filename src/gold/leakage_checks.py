from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd


DEFAULT_BANNED_COLUMNS = {
    "final_position",
    "position_final",
    "points",
    "duration",
    "number_of_laps",
    "classified_laps",
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
}


def run_gold_leakage_checks(
    features: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    feature_contract: dict | None = None,
) -> dict:
    """Run strict Gold leakage checks against feature and label contracts."""
    feature_columns = set(features.columns)
    label_columns = set(labels.columns) if labels is not None else set()
    key_columns = {"meeting_key", "session_key", "driver_number", "lap_number"}

    banned_in_features = sorted(feature_columns & DEFAULT_BANNED_COLUMNS)
    nullable_keys = {
        col: int(features[col].isna().sum())
        for col in ["session_key", "driver_number", "lap_number"]
        if col in features.columns and int(features[col].isna().sum()) > 0
    }
    duplicate_key_rows = 0
    if {"session_key", "driver_number", "lap_number"}.issubset(feature_columns):
        duplicate_key_rows = int(features.duplicated(["session_key", "driver_number", "lap_number"]).sum())

    registered_allowed = set()
    missing_contract_sections = []
    if feature_contract:
        for name, contract in feature_contract.get("feature_tables", {}).items():
            if not contract:
                missing_contract_sections.append(name)
                continue
            registered_allowed.update(contract.get("model_allowed_columns", []))

    non_feature_columns = key_columns | {
        "lap_start_time",
        "event_type",
        "session_name",
        "year",
        "circuit_short_name",
        "country_name",
        "full_name",
        "name_acronym",
        "team_name",
        "country_code",
        "scheduled_laps_source",
    }
    unchecked_model_columns = sorted(
        col for col in feature_columns - non_feature_columns if feature_contract and col not in registered_allowed
    )

    label_key_duplicates = 0
    nullable_targets = {}
    if labels is not None:
        if {"session_key", "driver_number"}.issubset(label_columns):
            label_key_duplicates = int(labels.duplicated(["session_key", "driver_number"]).sum())
        for col in ["target_finish_bucket", "target_win", "target_podium"]:
            if col in labels.columns:
                null_count = int(labels[col].isna().sum())
                if null_count:
                    nullable_targets[col] = null_count

    status = "PASS"
    blockers = []
    if banned_in_features:
        blockers.append("banned_columns_in_feature_table")
    if duplicate_key_rows:
        blockers.append("duplicate_feature_keys")
    if nullable_keys:
        blockers.append("nullable_feature_keys")
    if label_key_duplicates:
        blockers.append("duplicate_label_keys")
    if nullable_targets:
        blockers.append("nullable_targets")
    if blockers:
        status = "FAIL"

    return {
        "generated_at": datetime.now().isoformat(),
        "status": status,
        "blockers": blockers,
        "banned_columns_in_features": banned_in_features,
        "duplicate_feature_key_rows": duplicate_key_rows,
        "nullable_feature_keys": nullable_keys,
        "duplicate_label_key_rows": label_key_duplicates,
        "nullable_targets": nullable_targets,
        "missing_contract_sections": missing_contract_sections,
        "unchecked_model_columns": unchecked_model_columns,
    }


def write_gold_leakage_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

from __future__ import annotations

import pandas as pd

from src.gold.horizon_builder import build_horizon_datasets


LABEL_COLUMNS = [
    "target_win",
    "target_podium",
    "target_top10",
    "target_points",
    "target_dnf",
    "target_finish_bucket",
    "target_finish_bucket_label",
]


def build_training_datasets(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Join labels to Gold features and create target-specific training datasets."""
    feature_keys = {"session_key", "driver_number", "lap_number"}
    label_keys = {"session_key", "driver_number"}
    missing_features = sorted(feature_keys - set(features.columns))
    missing_labels = sorted(label_keys - set(labels.columns))
    if missing_features:
        raise ValueError(f"features is missing required columns: {missing_features}")
    if missing_labels:
        raise ValueError(f"labels is missing required columns: {missing_labels}")

    label_cols = ["session_key", "driver_number"] + [col for col in LABEL_COLUMNS if col in labels.columns]
    frame = features.merge(labels[label_cols], on=["session_key", "driver_number"], how="inner")
    frame = frame.dropna(subset=["target_finish_bucket"]).copy()
    frame["target_finish_bucket"] = pd.to_numeric(frame["target_finish_bucket"], errors="coerce").astype("int8")
    return build_horizon_datasets(frame)

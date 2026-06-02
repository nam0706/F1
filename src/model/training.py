from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import load_config
from src.utils import ensure_dir, utc_now_iso, write_csv, write_json, write_markdown


CLASS_LABELS = {
    0: "WIN",
    1: "PODIUM_NON_WIN",
    2: "POINTS_NON_PODIUM",
    3: "CLASSIFIED_OUTSIDE_POINTS",
    4: "DNF_DNS_DSQ_UNCLASSIFIED",
}

NON_MODEL_COLUMNS = {
    "target_win",
    "target_podium",
    "target_top10",
    "target_points",
    "target_dnf",
    "target_finish_bucket",
    "target_finish_bucket_label",
}


@dataclass(frozen=True)
class GoldModelPaths:
    training_dataset_path: Path
    feature_contract_path: Path
    leakage_report_path: Path
    model_dir: Path
    reports_dir: Path
    predictions_dir: Path
    artifact_path: Path
    metadata_path: Path
    holdout_predictions_path: Path
    model_report_path: Path
    confusion_matrix_path: Path
    feature_importance_path: Path


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_paths(config) -> GoldModelPaths:
    model_cfg = dict(config.raw.get("model", {}))
    root = config.project_root
    model_dir = _project_path(root, model_cfg.get("model_dir", "models"))
    reports_dir = _project_path(root, model_cfg.get("reports_dir", "reports/model"))
    predictions_dir = _project_path(root, model_cfg.get("predictions_dir", "data/gold/model"))
    return GoldModelPaths(
        training_dataset_path=_project_path(root, model_cfg.get("training_dataset_path", "data/gold/training/finish_bucket_live_any_lap.parquet")),
        feature_contract_path=_project_path(root, model_cfg.get("feature_contract_path", "data/gold/metadata/feature_contract.json")),
        leakage_report_path=_project_path(root, model_cfg.get("leakage_report_path", "data/gold/metadata/leakage_report.json")),
        model_dir=model_dir,
        reports_dir=reports_dir,
        predictions_dir=predictions_dir,
        artifact_path=_project_path(root, model_cfg.get("artifact_path", model_dir / "gold_finish_bucket_model.joblib")),
        metadata_path=_project_path(root, model_cfg.get("metadata_path", model_dir / "gold_model_metadata.json")),
        holdout_predictions_path=_project_path(root, model_cfg.get("holdout_predictions_path", predictions_dir / "holdout_predictions.parquet")),
        model_report_path=_project_path(root, model_cfg.get("model_report_path", reports_dir / "gold_model_report.md")),
        confusion_matrix_path=_project_path(root, model_cfg.get("confusion_matrix_path", reports_dir / "gold_confusion_matrix.csv")),
        feature_importance_path=_project_path(root, model_cfg.get("feature_importance_path", reports_dir / "gold_feature_importance.csv")),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_allowed_columns(feature_contract: dict[str, Any], dataset_columns: set[str]) -> list[str]:
    allowed: list[str] = []
    for contract in feature_contract.get("feature_tables", {}).values():
        for column in contract.get("model_allowed_columns", []):
            if column in dataset_columns and column not in NON_MODEL_COLUMNS and column not in allowed:
                allowed.append(column)
    if not allowed:
        raise ValueError("No model-allowed columns were found in feature_contract.json")
    return allowed


def _session_holdout_mask(df: pd.DataFrame, test_size: float) -> pd.Series:
    session_order = (
        df.groupby("session_key", as_index=False)["lap_start_time"]
        .min()
        .sort_values(["lap_start_time", "session_key"])
    )
    n_test = max(1, int(np.ceil(len(session_order) * test_size)))
    test_sessions = set(session_order.tail(n_test)["session_key"].tolist())
    return df["session_key"].isin(test_sessions)


def _build_pipeline(df: pd.DataFrame, feature_columns: list[str], model_cfg: dict[str, Any]) -> Pipeline:
    categorical_columns = [
        column
        for column in feature_columns
        if str(df[column].dtype) in {"object", "category", "string"} or column in {"year", "driver_number"}
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=model_cfg.get("max_depth", 16),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 20)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=int(model_cfg.get("random_state", 42)),
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{idx}" for idx in range(len(model.named_steps["classifier"].feature_importances_))]


def _write_holdout_predictions(
    df: pd.DataFrame,
    test_mask: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray,
    output_path: Path,
    compression: str | None,
) -> None:
    context_columns = [
        column
        for column in [
            "session_key",
            "driver_number",
            "lap_number",
            "lap_start_time",
            "event_type",
            "year",
            "circuit_short_name",
            "full_name",
            "name_acronym",
            "team_name",
            "current_position",
            "compound",
            "target_finish_bucket",
            "target_finish_bucket_label",
        ]
        if column in df.columns
    ]
    output = df.loc[test_mask, context_columns].copy().reset_index(drop=True)
    output["prediction"] = y_pred
    output["prediction_label"] = output["prediction"].map(CLASS_LABELS)
    output["correct"] = output["prediction"].eq(y_true.reset_index(drop=True))
    for class_id, label in CLASS_LABELS.items():
        output[f"proba_{class_id}_{label}"] = proba[:, class_id] if class_id < proba.shape[1] else np.nan
    ensure_dir(output_path.parent)
    output.to_parquet(output_path, index=False, compression=compression)


def _write_report(metadata: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Gold Finish-Bucket Model Report",
        "",
        f"Generated: {metadata['generated_at']}",
        "",
        "## Dataset",
        "",
        f"- Training dataset: `{metadata['training_dataset_path']}`",
        f"- Rows: {metadata['rows']:,}",
        f"- Feature columns: {metadata['feature_count']:,}",
        f"- Train sessions: {metadata['train_sessions']:,}",
        f"- Test sessions: {metadata['test_sessions']:,}",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {metadata['metrics']['accuracy']:.4f}",
        f"- Balanced accuracy: {metadata['metrics']['balanced_accuracy']:.4f}",
        f"- Macro F1: {metadata['metrics']['f1_macro']:.4f}",
        f"- Weighted F1: {metadata['metrics']['f1_weighted']:.4f}",
        f"- Log loss: {metadata['metrics']['log_loss']:.4f}",
        "",
        "## Leakage Contract",
        "",
        f"- Leakage report status: `{metadata['leakage_status']}`",
        f"- Feature contract: `{metadata['feature_contract_path']}`",
        "",
        "## Notes",
        "",
        "- Split is session-level, so laps from the same race are not shared between train and test.",
        "- Feature selection is loaded from Gold `feature_contract.json`.",
        "- Labels are joined only in Gold training datasets, not in `master_lap_features.parquet`.",
    ]
    write_markdown("\n".join(lines) + "\n", report_path)


def train_gold_finish_bucket_model(config_path: str | Path | None = None, dry_run: bool | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    model_cfg = dict(config.raw.get("model", {}))
    dry_run = config.dry_run_default if dry_run is None else dry_run
    paths = _resolve_paths(config)
    if dry_run:
        return {"status": "dry_run", "artifact_path": str(paths.artifact_path)}

    leakage_report = _read_json(paths.leakage_report_path)
    if leakage_report.get("status") != "PASS":
        raise RuntimeError(f"Gold leakage report is not PASS: {leakage_report}")

    feature_contract = _read_json(paths.feature_contract_path)
    df = pd.read_parquet(paths.training_dataset_path)
    target_column = str(model_cfg.get("target_column", "target_finish_bucket"))
    if target_column not in df.columns:
        raise ValueError(f"Training dataset is missing target column: {target_column}")

    feature_columns = _model_allowed_columns(feature_contract, set(df.columns))
    df = df.dropna(subset=["session_key", "driver_number", "lap_number", target_column]).copy()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce").astype("int64")
    test_size = float(model_cfg.get("test_size", 0.2))
    test_mask = _session_holdout_mask(df, test_size)
    train_mask = ~test_mask

    x_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, target_column]
    x_test = df.loc[test_mask, feature_columns]
    y_test = df.loc[test_mask, target_column]
    if x_train.empty or x_test.empty:
        raise ValueError("Session holdout produced an empty train or test split")

    model = _build_pipeline(df, feature_columns, model_cfg)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    proba = model.predict_proba(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "log_loss": float(log_loss(y_test, proba, labels=sorted(CLASS_LABELS))),
    }

    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=sorted(CLASS_LABELS)),
        index=[CLASS_LABELS[idx] for idx in sorted(CLASS_LABELS)],
        columns=[CLASS_LABELS[idx] for idx in sorted(CLASS_LABELS)],
    )
    feature_importance = pd.DataFrame(
        {
            "feature": _feature_names(model),
            "importance": model.named_steps["classifier"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    ensure_dir(paths.model_dir)
    ensure_dir(paths.reports_dir)
    ensure_dir(paths.predictions_dir)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "class_labels": CLASS_LABELS,
            "feature_contract_path": str(paths.feature_contract_path.relative_to(config.project_root)),
            "training_dataset_path": str(paths.training_dataset_path.relative_to(config.project_root)),
        },
        paths.artifact_path,
    )
    write_csv(cm.reset_index(names="actual"), paths.confusion_matrix_path)
    write_csv(feature_importance, paths.feature_importance_path)
    _write_holdout_predictions(df, test_mask, y_test, y_pred, proba, paths.holdout_predictions_path, config.gold_compression)

    metadata = {
        "status": "trained",
        "generated_at": utc_now_iso(),
        "training_dataset_path": str(paths.training_dataset_path.relative_to(config.project_root)),
        "feature_contract_path": str(paths.feature_contract_path.relative_to(config.project_root)),
        "leakage_status": leakage_report.get("status"),
        "artifact_path": str(paths.artifact_path.relative_to(config.project_root)),
        "metadata_path": str(paths.metadata_path.relative_to(config.project_root)),
        "holdout_predictions_path": str(paths.holdout_predictions_path.relative_to(config.project_root)),
        "rows": int(len(df)),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_sessions": int(df.loc[train_mask, "session_key"].nunique()),
        "test_sessions": int(df.loc[test_mask, "session_key"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "class_labels": CLASS_LABELS,
        "metrics": metrics,
        "classification_report": classification_report(y_test, y_pred, labels=sorted(CLASS_LABELS), target_names=list(CLASS_LABELS.values()), output_dict=True, zero_division=0),
    }
    write_json(metadata, paths.metadata_path)
    _write_report(metadata, paths.model_report_path)
    return metadata


if __name__ == "__main__":
    print(train_gold_finish_bucket_model())

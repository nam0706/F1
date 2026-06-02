from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.gold.feature_base import build_base_lap_frame, build_master_lap_features, write_feature_contract
from src.gold.interval_features import (
    build_interval_feature_contract,
    build_interval_lap_features,
    write_interval_feature_contract,
)
from src.gold.labels import write_race_result_labels
from src.gold.leakage_checks import run_gold_leakage_checks, write_gold_leakage_report
from src.gold.overtake_features import (
    build_overtake_feature_contract,
    build_overtake_lap_features,
    write_overtake_feature_contract,
)
from src.gold.paths import GoldPaths
from src.gold.pit_features import build_pit_feature_contract, build_pit_lap_features, write_pit_feature_contract
from src.gold.position_features import (
    build_position_feature_contract,
    build_position_lap_features,
    write_position_feature_contract,
)
from src.gold.stint_features import build_stint_feature_contract, build_stint_lap_features, write_stint_feature_contract
from src.gold.training_datasets import build_training_datasets
from src.gold.weather_features import (
    build_weather_feature_contract,
    build_weather_lap_features,
    write_weather_feature_contract,
)


def build_gold(config_path: str | Path | None = None, dry_run: bool | None = None) -> dict:
    """Build the Gold layer.

    This is currently a spec-first skeleton. Implementation will be filled module by module.
    """
    config = load_config(config_path)
    dry_run = config.dry_run_default if dry_run is None else dry_run
    gold_paths = GoldPaths(config.project_root / "data" / "gold")
    if not dry_run:
        gold_paths.ensure()

    session_result_path = config.cleaned_dir / "session_result.parquet"
    laps_path = config.cleaned_dir / "laps.parquet"
    sessions_path = config.cleaned_dir / "sessions.parquet"
    drivers_path = config.cleaned_dir / "drivers.parquet"
    starting_grid_path = config.cleaned_dir / "starting_grid.parquet"
    position_path = config.cleaned_dir / "position.parquet"
    pit_path = config.cleaned_dir / "pit.parquet"
    stints_path = config.cleaned_dir / "stints.parquet"
    intervals_path = config.cleaned_dir / "intervals.parquet"
    weather_path = config.cleaned_dir / "weather.parquet"
    overtakes_path = config.cleaned_dir / "overtakes.parquet"
    if not session_result_path.exists():
        raise FileNotFoundError("Missing Silver artifact: data/cleaned/session_result.parquet")
    for path in [laps_path, sessions_path, drivers_path, starting_grid_path, position_path, pit_path, stints_path, intervals_path, weather_path, overtakes_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing Silver artifact: {path}")

    session_result = pd.read_parquet(session_result_path)
    laps = pd.read_parquet(laps_path)
    sessions = pd.read_parquet(sessions_path)
    drivers = pd.read_parquet(drivers_path)
    starting_grid = pd.read_parquet(starting_grid_path)
    position = pd.read_parquet(position_path)
    pit = pd.read_parquet(pit_path)
    stints = pd.read_parquet(stints_path)
    intervals = pd.read_parquet(intervals_path)
    weather = pd.read_parquet(weather_path)
    overtakes = pd.read_parquet(overtakes_path)
    label_contract = None
    feature_contract = None
    position_contract = None
    pit_contract = None
    stint_contract = None
    interval_contract = None
    weather_contract = None
    overtake_contract = None
    leakage_report = None
    training_contract = None
    if not dry_run:
        label_contract = write_race_result_labels(
            session_result,
            gold_paths.labels_dir / "race_result_labels.parquet",
            gold_paths.metadata_dir / "label_contract.json",
        )
        base_lap_frame, feature_contract = build_base_lap_frame(laps, sessions, drivers, starting_grid)
        position_lap_features = build_position_lap_features(base_lap_frame, position)
        position_contract = build_position_feature_contract(position_lap_features)
        position_lap_features.to_parquet(
            gold_paths.features_dir / "position_lap_features.parquet",
            index=False,
            compression=config.gold_compression,
        )
        write_position_feature_contract(position_contract, gold_paths.metadata_dir / "position_feature_contract.json")

        pit_lap_features = build_pit_lap_features(base_lap_frame, pit)
        pit_contract = build_pit_feature_contract(pit_lap_features)
        pit_lap_features.to_parquet(
            gold_paths.features_dir / "pit_lap_features.parquet",
            index=False,
            compression=config.gold_compression,
        )
        write_pit_feature_contract(pit_contract, gold_paths.metadata_dir / "pit_feature_contract.json")

        stint_lap_features = build_stint_lap_features(base_lap_frame, stints)
        stint_contract = build_stint_feature_contract(stint_lap_features)
        stint_lap_features.to_parquet(
            gold_paths.features_dir / "stint_lap_features.parquet",
            index=False,
            compression=config.gold_compression,
        )
        write_stint_feature_contract(stint_contract, gold_paths.metadata_dir / "stint_feature_contract.json")

        interval_lap_features = build_interval_lap_features(base_lap_frame, intervals)
        interval_contract = build_interval_feature_contract(interval_lap_features)
        interval_lap_features.to_parquet(
            gold_paths.features_dir / "interval_lap_features.parquet",
            index=False,
            compression=config.gold_compression,
        )
        write_interval_feature_contract(interval_contract, gold_paths.metadata_dir / "interval_feature_contract.json")

        weather_lap_features = build_weather_lap_features(base_lap_frame, weather)
        weather_contract = build_weather_feature_contract(weather_lap_features)
        weather_lap_features.to_parquet(
            gold_paths.features_dir / "weather_lap_features.parquet",
            index=False,
            compression=config.gold_compression,
        )
        write_weather_feature_contract(weather_contract, gold_paths.metadata_dir / "weather_feature_contract.json")

        overtake_lap_features = build_overtake_lap_features(base_lap_frame, overtakes)
        overtake_contract = build_overtake_feature_contract(overtake_lap_features)
        overtake_lap_features.to_parquet(
            gold_paths.features_dir / "overtake_lap_features.parquet",
            index=False,
            compression=config.gold_compression,
        )
        write_overtake_feature_contract(overtake_contract, gold_paths.metadata_dir / "overtake_feature_contract.json")

        master_lap_features = build_master_lap_features({"base_lap_frame": base_lap_frame})
        for feature_table in [
            position_lap_features,
            pit_lap_features,
            stint_lap_features,
            interval_lap_features,
            weather_lap_features,
            overtake_lap_features,
        ]:
            master_lap_features = master_lap_features.merge(
                feature_table,
                on=["session_key", "driver_number", "lap_number"],
                how="left",
            )
        full_feature_contract = dict(feature_contract)
        full_feature_contract["columns"] = list(master_lap_features.columns)
        full_feature_contract["rows"] = int(len(master_lap_features))
        full_feature_contract["feature_tables"] = {
            "base": {
                "generated_at": feature_contract.get("generated_at"),
                "artifact": feature_contract.get("artifact"),
                "grain": feature_contract.get("grain"),
                "rows": feature_contract.get("rows"),
                "columns": feature_contract.get("columns"),
                "availability_rule": feature_contract.get("availability_rule"),
                "model_allowed_columns": feature_contract.get("model_allowed_columns", []),
            },
            "position": position_contract,
            "pit": pit_contract,
            "stint": stint_contract,
            "interval": interval_contract,
            "weather": weather_contract,
            "overtake": overtake_contract,
        }
        master_lap_features.to_parquet(
            gold_paths.features_dir / "master_lap_features.parquet",
            index=False,
            compression=config.gold_compression,
        )
        write_feature_contract(full_feature_contract, gold_paths.metadata_dir / "feature_contract.json")
        feature_contract = full_feature_contract

        labels = pd.read_parquet(gold_paths.labels_dir / "race_result_labels.parquet")
        leakage_report = run_gold_leakage_checks(master_lap_features, labels, feature_contract)
        write_gold_leakage_report(leakage_report, gold_paths.metadata_dir / "leakage_report.json")

        training_datasets = build_training_datasets(master_lap_features, labels)
        training_contract = {
            "datasets": {},
            "target": "target_finish_bucket",
            "horizons": ["live_any_lap", "horizon_25", "horizon_50", "horizon_75", "horizon_panel"],
            "leakage_report_status": leakage_report["status"],
        }
        output_names = {
            "live_any_lap": "finish_bucket_live_any_lap.parquet",
            "horizon_25": "finish_bucket_horizon_25.parquet",
            "horizon_50": "finish_bucket_horizon_50.parquet",
            "horizon_75": "finish_bucket_horizon_75.parquet",
            "horizon_panel": "finish_bucket_horizon_panel.parquet",
        }
        for key, dataset in training_datasets.items():
            output_path = gold_paths.training_dir / output_names[key]
            dataset.to_parquet(output_path, index=False, compression=config.gold_compression)
            training_contract["datasets"][key] = {
                "artifact": str(output_path.relative_to(config.project_root)),
                "rows": int(len(dataset)),
                "columns": int(len(dataset.columns)),
                "unique_sessions": int(dataset["session_key"].nunique()) if "session_key" in dataset.columns else 0,
                "unique_driver_sessions": int(dataset[["session_key", "driver_number"]].drop_duplicates().shape[0])
                if {"session_key", "driver_number"}.issubset(dataset.columns)
                else 0,
            }
        (gold_paths.metadata_dir / "training_dataset_contract.json").write_text(
            __import__("json").dumps(training_contract, indent=2, default=str),
            encoding="utf-8",
        )

    return {
        "status": "gold_built" if not dry_run else "dry_run",
        "gold_root": str(gold_paths.root),
        "dry_run": dry_run,
        "labels": label_contract,
        "features": feature_contract,
        "position_features": position_contract,
        "pit_features": pit_contract,
        "stint_features": stint_contract,
        "interval_features": interval_contract,
        "weather_features": weather_contract,
        "overtake_features": overtake_contract,
        "leakage_report": leakage_report,
        "training_datasets": training_contract,
    }


if __name__ == "__main__":
    print(build_gold())

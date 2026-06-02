# Gold Data Architecture

## Purpose

The Gold layer converts cleaned Silver facts into model-ready, temporally valid analytical datasets for live in-race F1 prediction.

The core product is a lap-level feature matrix that can support predictions at any lap and at fixed race-progress horizons. Gold must be reusable across multiple tasks: finish bucket, podium, top-10, DNF risk, tyre cliff, pit strategy, and overtake opportunity.

## Canonical Layout

```text
data/gold/
├── labels/
│   └── race_result_labels.parquet
├── features/
│   ├── master_lap_features.parquet
│   ├── pit_lap_features.parquet
│   ├── stint_lap_features.parquet
│   ├── interval_lap_features.parquet
│   ├── position_lap_features.parquet
│   ├── weather_session_features.parquet
│   ├── overtake_session_features.parquet
│   └── telemetry_lap_features.parquet
├── training/
│   ├── finish_bucket_live_any_lap.parquet
│   ├── finish_bucket_horizon_25.parquet
│   ├── finish_bucket_horizon_50.parquet
│   ├── finish_bucket_horizon_75.parquet
│   └── finish_bucket_horizon_panel.parquet
└── metadata/
    ├── gold_build_manifest.json
    ├── feature_contract.json
    ├── label_contract.json
    ├── leakage_report.json
    └── join_keys.json
```

## Artifact Ownership

| Artifact | Grain | Owner Module | Purpose |
|---|---:|---|---|
| `race_result_labels.parquet` | driver-session | `labels.py` | Finish outcome targets |
| `master_lap_features.parquet` | driver-session-lap | `feature_base.py` | Canonical non-telemetry feature table |
| `pit_lap_features.parquet` | driver-session-lap | `pit_features.py` | Pit state up to current lap |
| `stint_lap_features.parquet` | driver-session-lap | `stint_features.py` | Tyre compound and tyre age |
| `interval_lap_features.parquet` | driver-session-lap | `interval_features.py` | Gap/DRS-train context |
| `position_lap_features.parquet` | driver-session-lap | `position_features.py` | Position state and deltas |
| `weather_session_features.parquet` | session | `weather_features.py` | Race weather context |
| `overtake_session_features.parquet` | driver-session | `overtake_features.py` | Racecraft summary available up to horizon when possible |
| `telemetry_lap_features.parquet` | driver-session-lap | `telemetry_features.py` | Optional telemetry aggregates |

## Dataset Strategy

Gold will not use one huge always-joined table as the canonical source. The canonical approach is:

1. Build separate endpoint feature tables.
2. Build `master_lap_features.parquet` from non-telemetry features.
3. Build `telemetry_lap_features.parquet` separately.
4. Build training datasets by joining labels and selected optional feature tables.

This keeps telemetry optional, controls memory, and makes leakage audits tractable.

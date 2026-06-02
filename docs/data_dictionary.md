# Data Dictionary

## Gold Artifacts

| File | Grain | Primary Key | Purpose |
|---|---|---|---|
| `data/gold/features/master_lap_features.parquet` | driver-lap | `session_key + driver_number + lap_number` | Canonical lap-grain feature table |
| `data/gold/labels/race_result_labels.parquet` | driver-session | `session_key + driver_number` | Isolated post-race labels |
| `data/gold/training/finish_bucket_live_any_lap.parquet` | driver-lap | `session_key + driver_number + lap_number` | Live lap-slider training view |
| `data/gold/training/finish_bucket_horizon_25.parquet` | driver-session horizon | `session_key + driver_number` | 25% race-progress snapshot |
| `data/gold/training/finish_bucket_horizon_50.parquet` | driver-session horizon | `session_key + driver_number` | 50% race-progress snapshot |
| `data/gold/training/finish_bucket_horizon_75.parquet` | driver-session horizon | `session_key + driver_number` | 75% race-progress snapshot |
| `data/gold/training/finish_bucket_horizon_panel.parquet` | driver-session horizon panel | `session_key + driver_number + horizon` | Combined fixed-horizon panel |

## Metadata Contracts

| File | Purpose |
|---|---|
| `data/gold/metadata/feature_contract.json` | Model feature whitelist and availability rules |
| `data/gold/metadata/label_contract.json` | Label definitions and target distribution |
| `data/gold/metadata/leakage_report.json` | Leakage gate output |
| `data/gold/metadata/training_dataset_contract.json` | Training dataset row counts and horizon manifest |

## Model Artifacts

| File | Purpose |
|---|---|
| `models/gold_finish_bucket_model.joblib` | Serialized Gold-contract preprocessing and Random Forest classifier package |
| `models/gold_model_metadata.json` | Feature columns, split metadata, metrics, and class labels |
| `reports/model/gold_model_report.md` | Human-readable model report |
| `reports/model/gold_confusion_matrix.csv` | Session-holdout confusion matrix |
| `reports/model/gold_feature_importance.csv` | Random Forest feature importances after preprocessing |
| `data/gold/model/holdout_predictions.parquet` | Session-holdout predictions and class probabilities |

## Target

| Column | Description |
|---|---|
| `target_finish_bucket` | Multiclass final outcome: `0=WIN`, `1=PODIUM_NON_WIN`, `2=POINTS_NON_PODIUM`, `3=CLASSIFIED_OUTSIDE_POINTS`, `4=DNF_DNS_DSQ_UNCLASSIFIED` |

## Gold Feature Families

| Family | Representative Columns |
|---|---|
| Base/session context | `event_type`, `year`, `circuit_short_name`, `grid_position`, `race_progress_pct`, `laps_to_go` |
| Live position | `current_position`, `prev_position`, `position_delta_prev_lap`, `is_top3_live`, `is_top10_live` |
| Pit state | `pit_count_so_far`, `laps_since_last_pit`, `last_pit_duration` |
| Stint/tyre state | `stint_number`, `compound`, `current_tyre_age`, `laps_remaining_in_stint` |
| Interval/gap state | `interval_value_seconds`, `gap_to_leader_seconds`, `prev_gap_to_leader_seconds` |
| Weather | `track_temperature`, `air_temperature`, `humidity`, `pressure`, `rainfall`, `is_wet_track` |
| Racecraft | `prev_lap_overtakes_made`, `overtakes_made_so_far`, `net_overtakes_so_far` |

# Gold Layer Handoff

Generated: 2026-06-02 14:17:45 +07:00

## Files Modified or Created During This Turn

| File or Path | State |
|---|---|
| `src/gold/feature_base.py` | Added base feature whitelist metadata for strict leakage contracts. |
| `src/gold/position_features.py` | Added `position_change_direction` to the model-allowed contract. |
| `src/gold/stint_features.py` | Preserves all base lap rows and marks missing stint coverage instead of dropping unmatched laps. |
| `src/gold/interval_features.py` | Implemented time-safe interval/gap features at lap grain. |
| `src/gold/weather_features.py` | Implemented time-safe weather features at lap grain. |
| `src/gold/overtake_features.py` | Implemented timestamp-to-lap assignment and shifted overtake features so lap N uses only events before lap N. |
| `src/gold/horizon_builder.py` | Implemented `live_any_lap`, `25%`, `50%`, `75%`, and panel horizon dataset selection. |
| `src/gold/training_datasets.py` | Implemented label join and horizon dataset generation. |
| `src/gold/leakage_checks.py` | Implemented strict leakage audit and JSON report writer. |
| `src/gold/build_gold.py` | Integrated labels, feature modules, master feature table, leakage report, and training dataset outputs. |
| `data/gold/features/*.parquet` | Regenerated Gold feature artifacts. |
| `data/gold/labels/race_result_labels.parquet` | Regenerated driver-session finish labels. |
| `data/gold/training/*.parquet` | Created finish-bucket training datasets for live and fixed horizons. |
| `data/gold/metadata/*.json` | Regenerated feature, label, training, and leakage metadata contracts. |
| `scripts/build_gold_notebooks.py` | Created generator for Gold audit notebooks. |
| `eda/gold/notebooks/01_gold_label_audit.ipynb` | Created and executed Gold label audit notebook. |
| `eda/gold/notebooks/02_gold_feature_coverage.ipynb` | Created and executed Gold feature coverage notebook. |
| `eda/gold/notebooks/03_gold_leakage_audit.ipynb` | Created and executed Gold leakage audit notebook. |
| `eda/gold/notebooks/04_horizon_dataset_audit.ipynb` | Created and executed Gold horizon dataset audit notebook. |
| `eda/gold/notebooks/05_gold_final_summary.ipynb` | Created and executed Gold final summary notebook. |
| `eda/gold/notebooks/README.md` | Updated to document the new Gold notebook order and outputs. |
| `eda/gold/insights/README.md` | Updated to document generated Gold insights. |
| `eda/gold/outputs/charts/*` | Generated Plotly HTML charts from Gold notebooks. |
| `eda/gold/outputs/tables/*` | Generated compact CSV audit tables from Gold notebooks. |
| `eda/gold/outputs/reports/*` | Generated JSON audit reports from Gold notebooks. |
| `eda/gold/insights/01_gold_label_audit.md` | Generated label audit insights. |
| `eda/gold/insights/02_gold_feature_coverage.md` | Generated feature coverage insights. |
| `eda/gold/insights/03_gold_leakage_audit.md` | Generated leakage audit insights. |
| `eda/gold/insights/04_horizon_dataset_audit.md` | Generated horizon dataset insights. |
| `eda/gold/insights/05_gold_final_summary.md` | Generated final Gold readiness insights. |
| `app.py` | Deleted legacy Streamlit app; future app should be rebuilt on Gold outputs. |
| `src/model_training.py` | Deleted legacy model training code tied to old processed/app artifacts. |
| `src/feature_engineering.py` | Deleted legacy Gold builder after moving scheduled-lap constants into `src/gold/scheduled_laps.py`. |
| `src/gold/scheduled_laps.py` | Created schedule constants module used by Gold base features. |
| `run_e2e_pipeline.py` | Updated feature step to call `src.gold.build_gold.build_gold()` and removed active model/app stage. |
| `configs/pipeline_config.yaml` | Removed active `model` pipeline step and replaced old processed/app paths with Gold contract training metadata. |
| `requirements.txt` | Re-added `streamlit` for the new Gold demo app; Gemini package remains removed. |
| `README.md` | Rewritten for the current Gold medallion pipeline. |
| `docs/data_dictionary.md` | Rewritten for `data/gold/` artifacts and feature families. |
| `docs/pipeline_code_walkthrough.md` | Rewritten for the current crawl-clean-validate-feature pipeline. |
| `models/`, `notebook/`, `tools/` | Removed empty legacy directories. |
| `src/model/__init__.py` | Created Gold-contract model package. |
| `src/model/training.py` | Created model training code that loads feature whitelist from `feature_contract.json`. |
| `models/gold_finish_bucket_model.joblib` | Created trained Gold finish-bucket model artifact. |
| `models/gold_model_metadata.json` | Created model metadata and metrics artifact. |
| `reports/model/gold_model_report.md` | Created human-readable model report. |
| `reports/model/gold_confusion_matrix.csv` | Created holdout confusion matrix. |
| `reports/model/gold_feature_importance.csv` | Created model feature importance table. |
| `data/gold/model/holdout_predictions.parquet` | Created holdout predictions and probabilities. |
| `app.py` | Reworked into a thin launcher for the modular Gold dashboard package. |
| `src/dashboard/__init__.py` | Created dashboard package marker. |
| `src/dashboard/paths.py` | Created centralized dashboard artifact path registry. |
| `src/dashboard/data.py` | Created cached data/model loaders for the Streamlit app. |
| `src/dashboard/inference.py` | Created model probability helper. |
| `src/dashboard/charts.py` | Created Plotly chart builders. |
| `src/dashboard/pages/overview.py` | Created Gold overview page renderer. |
| `src/dashboard/pages/predictor.py` | Created race predictor page renderer. |
| `src/dashboard/main.py` | Created Streamlit app composition entrypoint. |
| `docs/mlops_app_handoff.md` | Rewritten as the current Gold layer handoff. |

## Current Exact State

Gold architecture is now implemented under:

- Code: `src/gold/`
- Data: `data/gold/`
- Labels: `data/gold/labels/race_result_labels.parquet`
- Master features: `data/gold/features/master_lap_features.parquet`
- Training datasets: `data/gold/training/`
- Contracts and audit reports: `data/gold/metadata/`

Canonical feature grain:

```text
session_key + driver_number + lap_number
```

Label grain:

```text
session_key + driver_number
```

Prediction timestamp contract:

```text
before lap N starts
```

## Artifact Metrics

| Artifact | Rows | Columns |
|---|---:|---:|
| `master_lap_features.parquet` | 63,676 | 58 |
| `position_lap_features.parquet` | 63,676 | 10 |
| `pit_lap_features.parquet` | 63,676 | 8 |
| `stint_lap_features.parquet` | 63,676 | 10 |
| `interval_lap_features.parquet` | 63,676 | 9 |
| `weather_lap_features.parquet` | 63,676 | 10 |
| `overtake_lap_features.parquet` | 63,676 | 8 |
| `race_result_labels.parquet` | 1,374 | 16 |
| `finish_bucket_live_any_lap.parquet` | 62,422 | 65 |
| `finish_bucket_horizon_25.parquet` | 1,328 | 67 |
| `finish_bucket_horizon_50.parquet` | 1,328 | 67 |
| `finish_bucket_horizon_75.parquet` | 1,328 | 67 |
| `finish_bucket_horizon_panel.parquet` | 3,984 | 67 |

## Leakage Audit State

Current leakage report:

```text
status: PASS
blockers: []
banned_columns_in_features: []
duplicate_feature_key_rows: 0
duplicate_label_key_rows: 0
nullable_targets: {}
unchecked_model_columns: []
```

Important implemented safeguards:

- Post-race label columns are isolated in `data/gold/labels/` and are joined only when building training datasets.
- Overtake event counts are shifted so current lap rows do not use same-lap events.
- Pit features use only pit events strictly before the current lap.
- Weather and interval features use the latest source event before current lap start.
- Feature contract now uses whitelist metadata, not only a blacklist.

## Known Data Quality Notes

- `stint_lap_features` keeps all 63,676 base lap rows.
- 156 lap rows have missing stint assignment, isolated to incomplete Silver stint coverage for `session_key=11240` first-stint rows.
- This is preserved as missing feature coverage instead of silently dropping lap rows.
- Scheduled laps use circuit/session mapping for 67 sessions and observed max-lap fallback for session `11286`.

## Verification Completed

Commands completed successfully:

```powershell
.\venv\Scripts\python.exe -m src.gold.build_gold
Get-ChildItem -Path 'src/gold' -Filter '*.py' | ForEach-Object { .\venv\Scripts\python.exe -m py_compile $_.FullName }
.\venv\Scripts\python.exe scripts/build_gold_notebooks.py
Get-ChildItem -Path 'eda/gold/notebooks' -Filter '*.ipynb' | Sort-Object Name | ForEach-Object { .\venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace $_.FullName }
.\venv\Scripts\python.exe -c "from run_e2e_pipeline import run_e2e_pipeline; run_e2e_pipeline(steps=['feature'])"
.\venv\Scripts\python.exe -c "from run_e2e_pipeline import run_e2e_pipeline; run_e2e_pipeline(steps=['model'])"
```

Final build summary:

```text
status: gold_built
feature_rows: 63676
feature_cols: 58
leakage_status: PASS
training live_any_lap rows: 62422
training horizon_25 rows: 1328
training horizon_50 rows: 1328
training horizon_75 rows: 1328
training horizon_panel rows: 3984
model status: trained
model features: 49
model train sessions: 54
model test sessions: 14
model accuracy: 0.7409
model balanced accuracy: 0.6191
model macro F1: 0.6092
```

Gold notebook execution completed successfully for:

```text
01_gold_label_audit.ipynb
02_gold_feature_coverage.ipynb
03_gold_leakage_audit.ipynb
04_horizon_dataset_audit.ipynb
05_gold_final_summary.ipynb
```

## Next Logical Work

1. Decide whether telemetry should remain separate Gold output or become optional joined features.
2. Add automated tests for leakage checks, horizon selection, and model feature whitelist loading.
3. Improve model selection beyond Random Forest after the demo app is validated.
4. Decide which app interactions should be kept for the final project UI.

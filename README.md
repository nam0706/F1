# F1 WinRate Predictor Data Pipeline

This repository focuses on the medallion data pipeline, Gold dataset contract,
Gold-compliant model training, and a lightweight Streamlit demo app for F1
finish-bucket prediction.

## Current Architecture

```text
OpenF1 / FastF1
      |
      v
data/raw/        Bronze raw API/session files
      |
      v
data/cleaned/    Silver typed and normalized tables
      |
      v
data/gold/       Gold labels, feature tables, training horizons, contracts
```

Current Gold artifacts:

```text
data/gold/labels/race_result_labels.parquet
data/gold/features/master_lap_features.parquet
data/gold/training/finish_bucket_live_any_lap.parquet
data/gold/training/finish_bucket_horizon_25.parquet
data/gold/training/finish_bucket_horizon_50.parquet
data/gold/training/finish_bucket_horizon_75.parquet
data/gold/training/finish_bucket_horizon_panel.parquet
data/gold/metadata/feature_contract.json
data/gold/metadata/label_contract.json
data/gold/metadata/leakage_report.json
models/gold_finish_bucket_model.joblib
models/gold_model_metadata.json
data/gold/model/holdout_predictions.parquet
```

Gold grain:

```text
session_key + driver_number + lap_number
```

Label grain:

```text
session_key + driver_number
```

## Run Commands

Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run configured pipeline:

```powershell
.\venv\Scripts\python.exe run_e2e_pipeline.py
```

Run selected steps:

```powershell
.\venv\Scripts\python.exe -c "from run_e2e_pipeline import run_e2e_pipeline; run_e2e_pipeline(steps=['clean','validate','feature'])"
.\venv\Scripts\python.exe -c "from run_e2e_pipeline import run_e2e_pipeline; run_e2e_pipeline(steps=['feature'])"
.\venv\Scripts\python.exe -c "from run_e2e_pipeline import run_e2e_pipeline; run_e2e_pipeline(steps=['model'])"
```

Launch the demo app:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Build and execute Gold EDA notebooks:

```powershell
.\venv\Scripts\python.exe scripts/build_gold_notebooks.py
Get-ChildItem -Path 'eda/gold/notebooks' -Filter '*.ipynb' | Sort-Object Name | ForEach-Object { .\venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace $_.FullName }
```

Compile-check Python code:

```powershell
Get-ChildItem -Path 'src' -Filter '*.py' -Recurse | ForEach-Object { .\venv\Scripts\python.exe -m py_compile $_.FullName }
.\venv\Scripts\python.exe -m py_compile run_e2e_pipeline.py scripts/build_gold_notebooks.py
```

## Current Gold Metrics

```text
master_lap_features.parquet: 63,676 rows x 58 columns
race_result_labels.parquet: 1,374 rows x 16 columns
finish_bucket_live_any_lap.parquet: 62,422 rows
finish_bucket_horizon_25.parquet: 1,328 rows
finish_bucket_horizon_50.parquet: 1,328 rows
finish_bucket_horizon_75.parquet: 1,328 rows
finish_bucket_horizon_panel.parquet: 3,984 rows
leakage_report: PASS
unchecked_model_columns: []
gold model macro F1: 0.6092
```

## Leakage Boundary

Gold owns feature construction, rolling/shift logic, label isolation, horizon
selection, and leakage metadata. Model training must load feature columns from
`data/gold/metadata/feature_contract.json`.

Post-race outcome columns are labels only and must not appear in
`master_lap_features.parquet`.

## EDA Workspace

Layered EDA assets live under:

```text
eda/bronze/
eda/silver/
eda/gold/
```

Gold notebooks audit generated Gold artifacts only. They do not read raw data
and do not rebuild feature logic.

## Demo App

`app.py` is now a thin launcher for the modular Streamlit package under
`src/dashboard/`. The app reads only Gold artifacts and the Gold-contract model:

- `data/gold/training/finish_bucket_live_any_lap.parquet`
- `data/gold/metadata/feature_contract.json`
- `models/gold_finish_bucket_model.joblib`

The app has two tabs:

- `Gold Overview`: label balance, model metrics, leakage status, horizon summary.
- `Race Predictor`: session/driver/lap selector with finish-bucket probabilities.

Dashboard modules:

```text
src/dashboard/paths.py
src/dashboard/data.py
src/dashboard/inference.py
src/dashboard/charts.py
src/dashboard/pages/overview.py
src/dashboard/pages/predictor.py
src/dashboard/main.py
```

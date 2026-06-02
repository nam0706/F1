# Pipeline Code Walkthrough

This is the current pipeline contract after the Gold-layer migration.

## Layers

| Layer | Directory | Purpose |
|---|---|---|
| Bronze | `data/raw/` | Preserved raw OpenF1/FastF1 snapshot. |
| Silver | `data/cleaned/` | Typed, normalized, schema-aligned cleaned tables. |
| Gold | `data/gold/` | Labels, feature tables, horizon training datasets, and metadata contracts. |
| Model | `models/`, `reports/model/`, `data/gold/model/` | Gold-contract model artifact, reports, and holdout predictions. |
| EDA | `eda/` | Audit notebooks and generated charts/tables/reports. |
| Metadata | `data/metadata/`, `data/gold/metadata/` | Crawl, quality, feature, label, and leakage metadata. |

## Entry Point

Run the configured pipeline with:

```powershell
.\venv\Scripts\python.exe run_e2e_pipeline.py
```

Configured order:

```text
crawl -> clean -> validate -> feature -> model
```

The `feature` step calls:

```python
src.gold.build_gold.build_gold()
```

The `model` step calls:

```python
src.model.training.train_gold_finish_bucket_model()
```

The old app/model stage has been replaced by Gold-contract training and a new
demo app.

## Bronze

`src/crawler.py` writes raw endpoint files under `data/raw/` according to
`configs/pipeline_config.yaml`. Raw files are preserved and are not mutated by
Silver or Gold code.

## Silver

`src/clean_data.py` creates cleaned parquet tables under `data/cleaned/`.
Silver EDA reads cleaned outputs only.

## Gold

`src/gold/build_gold.py` creates:

```text
data/gold/labels/race_result_labels.parquet
data/gold/features/master_lap_features.parquet
data/gold/features/*_lap_features.parquet
data/gold/training/finish_bucket_*.parquet
data/gold/metadata/*.json
```

Key guarantees:

- Canonical feature grain is `session_key + driver_number + lap_number`.
- Label grain is `session_key + driver_number`.
- Labels stay isolated until training dataset construction.
- Feature selection is governed by `feature_contract.json`.
- `leakage_report.json` must pass before model training.
- Horizon datasets cover `live_any_lap`, `25%`, `50%`, `75%`, and a combined panel.

## Gold EDA

Gold audit notebooks live in:

```text
eda/gold/notebooks/
```

Execute them with:

```powershell
.\venv\Scripts\python.exe scripts/build_gold_notebooks.py
Get-ChildItem -Path 'eda/gold/notebooks' -Filter '*.ipynb' | Sort-Object Name | ForEach-Object { .\venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace $_.FullName }
```

These notebooks audit existing `data/gold/` artifacts only. They do not rebuild
features and do not read raw data.

## Model Training

`src/model/training.py` trains from:

```text
data/gold/training/finish_bucket_live_any_lap.parquet
```

It loads allowed model features from:

```text
data/gold/metadata/feature_contract.json
```

Outputs:

```text
models/gold_finish_bucket_model.joblib
models/gold_model_metadata.json
reports/model/gold_model_report.md
reports/model/gold_confusion_matrix.csv
reports/model/gold_feature_importance.csv
data/gold/model/holdout_predictions.parquet
```

## Demo App

`app.py` is a lightweight Streamlit demo that reads Gold artifacts and the new
Gold-contract model only. It provides a Gold overview tab and a race predictor
tab with session, driver, and lap controls.

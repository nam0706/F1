# Final Dataset Explorer

Streamlit dashboard for the final F1 lap-level dataset. The app is a data-quality and feature-exploration demo, not a prediction app.

## Run

From the project root:

```powershell
streamlit run apps/final_dataset_explorer/streamlit_app.py
```

## Inputs

- `data/processed/master_dataset.csv`
- `data/metadata/feature_engineering_metadata.json`
- `reports/data_quality/master_dataset_audit.csv`
- `reports/data_quality/cleaning_summary.csv`
- `data/cleaned/location.parquet` for the optional Race Replay tab

## Purpose

- Show final dataset size, grain, and prediction contract.
- Verify leakage-safe feature groups for `Live before lap N`.
- Explore coverage, missingness, targets, pace, tyre, weather, pit, race-control, telemetry, location, and overtakes.
- Demonstrate which columns are allowed for future model input and which columns are diagnostic/outcome only.
- Replay car locations for a selected session using cleaned FastF1 location coordinates.

The Race Replay tab is inspired by `IAmTomShaw/f1-race-replay`, but is implemented for this project as a Streamlit/Plotly data explorer over the cleaned `location.parquet` artifact.

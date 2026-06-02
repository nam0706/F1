# Gold EDA Notebooks

These notebooks audit generated `data/gold/` artifacts only. Core Gold build
logic lives in `src/gold/`; notebooks must not rebuild features or read raw
data.

## Notebook Order

1. `01_gold_label_audit.ipynb`
2. `02_gold_feature_coverage.ipynb`
3. `03_gold_leakage_audit.ipynb`
4. `04_horizon_dataset_audit.ipynb`
5. `05_gold_final_summary.ipynb`

## Outputs

- Charts: `eda/gold/outputs/charts/<notebook_name>/`
- Tables: `eda/gold/outputs/tables/<notebook_name>/`
- Reports: `eda/gold/outputs/reports/<notebook_name>/`
- Insights: `eda/gold/insights/`

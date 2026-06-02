# Gold Finish-Bucket Model Report

Generated: 2026-06-02T07:58:09.207883+00:00

## Dataset

- Training dataset: `data\gold\training\finish_bucket_live_any_lap.parquet`
- Rows: 62,422
- Feature columns: 49
- Train sessions: 54
- Test sessions: 14

## Metrics

- Accuracy: 0.7409
- Balanced accuracy: 0.6191
- Macro F1: 0.6092
- Weighted F1: 0.7224
- Log loss: 0.7436

## Leakage Contract

- Leakage report status: `PASS`
- Feature contract: `data\gold\metadata\feature_contract.json`

## Notes

- Split is session-level, so laps from the same race are not shared between train and test.
- Feature selection is loaded from Gold `feature_contract.json`.
- Labels are joined only in Gold training datasets, not in `master_lap_features.parquet`.

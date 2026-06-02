# Label Contract

## Label Grain

Labels are built at:

```text
session_key + driver_number
```

Each label row represents the final race outcome for one driver in one Race or Sprint session.

## Output

```text
data/gold/labels/race_result_labels.parquet
```

## Required Columns

| Column | Type | Description |
|---|---|---|
| `session_key` | integer | Race/Sprint session id |
| `driver_number` | integer | Driver number in that session |
| `final_position` | float/integer nullable | Final classified position if available |
| `classified_laps` | float/integer nullable | Laps completed from `session_result` |
| `target_win` | integer | 1 if final position is P1 |
| `target_podium` | integer | 1 if final position is P1-P3 |
| `target_top10` | integer | 1 if final position is P1-P10 |
| `target_points` | integer | 1 if points > 0 |
| `target_dnf` | integer | 1 if DNF/DNS/DSQ or no final position |
| `target_finish_bucket` | integer | Multi-class finish bucket |
| `label_source` | string | Source table, normally `session_result` |

## Finish Bucket Definition

Initial bucket policy:

| Bucket | Meaning |
|---:|---|
| `0` | Win |
| `1` | Podium non-win |
| `2` | Points non-podium |
| `3` | Classified outside points |
| `4` | DNF/DNS/DSQ/unclassified |

This bucket is intentionally race-result based, not lap-state based.

## Label Rules

1. Labels may use post-race outcome columns because labels are not features.
2. Label generation is isolated from feature generation.
3. Labels are joined only in `training_datasets.py`.
4. Any column used to construct a label must be banned as a model feature unless explicitly transformed into a time-valid feature.

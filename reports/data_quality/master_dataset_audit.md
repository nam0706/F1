# Master Dataset Audit

- Artifact: `data/processed/master_dataset.csv`
- Rows: 82,535
- Columns: 112
- Grain: `session_key + driver_number + lap_number`
- Prediction contract: `live_before_lap_n`
- Model-safe numeric features: 69
- Outcome columns at end: `final_position, target_win, target_podium, target_top10`

| check                              | status   | value                                                   | expected                                                   | severity   |
|:-----------------------------------|:---------|:--------------------------------------------------------|:-----------------------------------------------------------|:-----------|
| master_rows                        | pass     | 82535                                                   | 82535                                                      | info       |
| master_columns                     | pass     | 112                                                     | 112                                                        | info       |
| duplicate_grain_rows               | pass     | 0                                                       | 0                                                          | info       |
| outcomes_are_last_columns          | pass     | final_position, target_win, target_podium, target_top10 | final_position, target_win, target_podium, target_top10    | info       |
| model_feature_leakage_blocklist    | pass     | none                                                    | no labels/current-lap diagnostics in model_feature_columns | info       |
| metadata_feature_list_matches_code | pass     | metadata=69, code=69                                    | same model-safe feature set                                | info       |
| telemetry_prev_speed_coverage      | pass     | 71.24%                                                  | >30%                                                       | info       |
| location_prev_coverage             | pass     | 71.24%                                                  | >30%                                                       | info       |

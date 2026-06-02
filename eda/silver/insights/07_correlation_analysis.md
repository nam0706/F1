# Silver Correlation Analysis Insights

**Generated at:** 2026-06-02 00:46:47

## Key Observations

- Built a compact driver-session matrix with 1,374 rows and 55 columns.
- Reviewed 29 candidate model features.
- Strongest safe finish-position correlation: grid_pos = 0.742.

## Issues

- 21 high-correlation feature pairs need Gold feature selection review.
- Outcome-derived columns such as points and positions_gained are documented as EDA-only leakage risks.

## Recommendations

- Use grid, lap pace, stint, weather, racecraft, and sampled telemetry families as Gold inputs.
- Exclude labels and post-race outcome columns from training features.
- Resolve high-correlation pairs before model training to reduce redundant signal.

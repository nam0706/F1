# Gold Horizon Dataset Audit

**Generated at:** 2026-06-02 14:28:51

## Key Observations

- Fixed horizons keep one selected row per driver-session.
- Live-any-lap keeps many rows per driver-session for interactive prediction.
- Target distribution is expected to remain stable across fixed horizons.

## Issues

- None

## Recommendations

- Use fixed horizons for comparable race-progress experiments.
- Use live_any_lap for Streamlit-style lap slider prediction and larger training volume.

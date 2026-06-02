# Silver Telemetry Analysis Insights

**Generated at:** 2026-06-02 00:46:38

## Key Observations

- Sampled 400,000 car telemetry rows and 180,000 location rows.
- Aggregated 200 driver-session telemetry profiles.
- Strongest sampled telemetry/finish correlation: avg_rpm = -0.164.

## Issues

- None

## Recommendations

- Use full row-group aggregation in Gold for final telemetry features; this notebook only validates signal direction.
- Keep avg_speed, throttle, brake, DRS rate, and lap-type features as candidate telemetry features.
- Use location data carefully; sampled GPS coverage is useful for sanity checks but not final racing-line modeling.

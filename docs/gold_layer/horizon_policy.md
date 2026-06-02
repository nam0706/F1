# Horizon Policy

## Supported Horizons

Gold supports four prediction scopes:

```text
live_any_lap
25%
50%
75%
```

## Live Any Lap

`live_any_lap` keeps all eligible lap rows after minimal history requirements are satisfied.

Output:

```text
data/gold/training/finish_bucket_live_any_lap.parquet
```

Use case:

- Streamlit lap slider
- Live race simulation
- More training rows
- Model learns how uncertainty changes across race progress

## Fixed Race-Progress Horizons

Fixed horizons select one row per driver-session nearest to a race-progress percentage.

Outputs:

```text
data/gold/training/finish_bucket_horizon_25.parquet
data/gold/training/finish_bucket_horizon_50.parquet
data/gold/training/finish_bucket_horizon_75.parquet
data/gold/training/finish_bucket_horizon_panel.parquet
```

## Race Progress Calculation

```text
race_progress_pct = lap_number / scheduled_laps
```

Where possible, `scheduled_laps` should come from a circuit/session schedule map. If unavailable, it may be inferred conservatively from the maximum completed lap in the session, but the source must be recorded.

## Horizon Selection

For each `session_key + driver_number + horizon`, choose the row with minimum absolute distance:

```text
abs(race_progress_pct - horizon_pct)
```

Ties are resolved by choosing the earlier lap to avoid using later information.

## Minimum Eligibility

Rows are eligible only if:

1. `lap_number >= 2`
2. Required join keys are non-null
3. Required label exists
4. Feature timestamp is legal under the leakage policy

Optional stricter mode may require at least five previous laps for rolling features.

# Feature Contract

## Primary Feature Grain

The canonical Gold feature grain is:

```text
session_key + driver_number + lap_number
```

Each row represents the model state at a prediction point during a race.

## Feature Families

| Family | Example Features | Grain | Time Rule |
|---|---|---:|---|
| Race progress | `race_progress_pct`, `laps_to_go` | lap | Known from schedule/current lap |
| Lap pace history | `prev_lap_duration`, `rolling_avg_lap_5` | lap | Must use `shift(1)` |
| Position state | `current_position`, `prev_position`, `position_delta_3` | lap | State at or before prediction lap |
| Pit state | `pit_count_so_far`, `laps_since_last_pit`, `last_pit_duration` | lap | Past pit events only |
| Stint/tyre | `compound`, `current_tyre_age`, `stint_number` | lap | Current stint at prediction lap |
| Interval/gap | `gap_to_leader`, `interval_to_ahead`, `drs_train_flag` | lap | Latest known interval at/before lap |
| Weather | `track_temperature`, `rainfall`, `humidity` | session/time | Latest known weather at/before lap if timestamped; session aggregate otherwise |
| Overtake | `overtakes_made_so_far`, `net_overtakes_so_far` | lap/session | Prefer cumulative up to lap; session aggregate only for EDA |
| Driver/team/circuit context | historical rates, circuit class | pre-race/session | Must be historical or static |
| Telemetry aggregate | speed/throttle/brake/DRS rolling stats | lap | Separate optional artifact |

## Rolling and Shift Rules

Gold owns temporal features. Model training must not create F1 domain rolling features.

Correct pattern:

```python
prev_lap_duration = lap_duration.shift(1)
rolling_avg_lap_5 = lap_duration.shift(1).rolling(5).mean()
```

Incorrect pattern:

```python
rolling_avg_lap_5 = lap_duration.rolling(5).mean()
```

The incorrect version includes the current lap result and can leak information when predicting before lap completion.

## Join Keys

| Table | Join Keys |
|---|---|
| Labels | `session_key`, `driver_number` |
| Master lap features | `session_key`, `driver_number`, `lap_number` |
| Pit lap features | `session_key`, `driver_number`, `lap_number` |
| Stint lap features | `session_key`, `driver_number`, `lap_number` |
| Interval lap features | `session_key`, `driver_number`, `lap_number` |
| Position lap features | `session_key`, `driver_number`, `lap_number` |
| Weather session features | `session_key` |
| Overtake session features | `session_key`, `driver_number` |
| Telemetry lap features | `session_key`, `driver_number`, `lap_number` |

## Output Contract

All feature tables must include:

```text
session_key
driver_number
lap_number  # except session-level or driver-session-level feature tables
```

All generated features must include metadata in:

```text
data/gold/metadata/feature_contract.json
```

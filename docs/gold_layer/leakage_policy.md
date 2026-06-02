# Leakage Policy

This project uses a strict leakage policy because live in-race prediction is highly vulnerable to accidental future information.

## Core Principle

For a prediction row at lap `N`, a feature is allowed only if it would have been available at the prediction timestamp.

If availability is ambiguous, the feature is banned until proven safe.

## Prediction Timestamp

Default prediction timestamp:

```text
before lap N starts
```

Therefore, historical lap pace features must be shifted:

```text
lap N feature can use lap <= N-1 pace outcomes
```

For UI simulation, a separate "after lap N completed" mode can be created later, but it must be a different dataset contract.

## Strict Whitelist Rule

Gold features must be registered in `feature_contract.json` with:

```text
feature_name
source_table
grain
availability_rule
uses_current_lap_result
uses_future_lap_result
uses_post_race_outcome
model_allowed
```

A feature is model-allowed only when:

```text
uses_future_lap_result = false
uses_post_race_outcome = false
availability_rule is explicit
```

## Banned as Model Features

Always banned:

```text
final_position
position_final
points
duration
number_of_laps
target_win
target_podium
target_top10
target_points
target_dnf
target_finish_bucket
positions_gained_final
classified_laps
dnf
dns
dsq
```

Conditionally banned:

```text
current lap_duration at lap N
current sector durations at lap N
current speed traps at lap N
session-level aggregates computed using the full race
driver/team/circuit historical aggregates that include the current session outcome
```

## Allowed Examples

Allowed before lap `N`:

```text
prev_lap_duration
rolling_avg_lap_3 from shifted lap_duration
rolling_std_lap_5 from shifted lap_duration
current_position from latest known position before lap N
pit_count_so_far before lap N
laps_since_last_pit before lap N
current_tyre_age at lap N
compound at lap N
race_progress_pct
laps_to_go
latest weather observed before lap N
```

## Label Join Rule

Labels are joined only in training dataset construction. After label join:

1. `target_*` columns are labels only.
2. Label source columns remain banned.
3. Training code must select feature columns from the feature contract, not by dropping a small blacklist.

## Leakage Audit Outputs

Gold must produce:

```text
data/gold/metadata/leakage_report.json
```

The report must include:

- banned columns found in feature tables
- unregistered feature columns
- rolling features missing shift marker
- label columns present before training join
- nullable target rows
- horizon selection source

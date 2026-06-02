# Gold Layer Specification

This directory defines the Gold layer design for the F1 live in-race prediction project.

The Gold layer is the first layer where model-ready analytical features are created. It must preserve strict time boundaries: a row at lap `N` can only contain information available at or before the prediction timestamp for lap `N`.

## Decisions

| Topic | Decision |
|---|---|
| Gold folder | `data/gold/` |
| Code location | `src/gold/` |
| Primary prediction grain | `session_key + driver_number + lap_number` |
| Label grain | `session_key + driver_number` |
| Horizons | `live_any_lap`, `25%`, `50%`, `75%` |
| Main target | `target_finish_bucket` |
| Approach | Spec-first, code-after |

## Documents

- [gold_data_architecture.md](gold_data_architecture.md): canonical Gold layout and artifact ownership.
- [label_contract.md](label_contract.md): race result labels and target definitions.
- [feature_contract.md](feature_contract.md): feature families, grains, join keys, and allowed time semantics.
- [horizon_policy.md](horizon_policy.md): live and fixed-race-progress dataset rules.
- [leakage_policy.md](leakage_policy.md): strict leakage controls and banned columns.
- [model_boundary.md](model_boundary.md): responsibilities split between Gold and model training.

## Non-Negotiable Rules

1. Silver and Gold EDA must never read from `data/raw/`.
2. Labels are built separately from features.
3. Labels are joined only when creating training datasets.
4. Rolling features must use shifted history.
5. Outcome-derived columns are never model features.
6. Telemetry remains a separate optional Gold artifact unless explicitly joined for an experiment.

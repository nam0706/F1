# Gold Label Audit

**Generated at:** 2026-06-02 14:28:24

## Key Observations

- Gold labels contain 1,374 driver-session outcomes across 68 sessions.
- Target classes are intentionally imbalanced because F1 sporting outcomes are constrained by grid size and points rules.
- Post-race outcome fields remain isolated in the label artifact and are not present in master_lap_features.

## Issues

- None

## Recommendations

- Use stratified validation where possible because finish buckets are imbalanced.
- Keep label columns out of feature selection; join them only in training dataset generation.

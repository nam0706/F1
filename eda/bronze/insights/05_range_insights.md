# Range Validation Insights

**Generated at:** 2026-06-01 18:01:22

## Summary

Validated numeric range rules across 15 endpoints.

## Key Observations

- Columns checked: 12
- Non-blocking range warnings: 4

## Issues

- None

## Recommendations

- Use min_violation, max_violation, and sample_values columns to decide Silver mitigation.
- Treat weather pressure and lap-duration violations as contextual warnings unless confirmed impossible.

## Next Steps

- Run 06_temporal_checks.ipynb

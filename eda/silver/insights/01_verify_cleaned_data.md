# Silver Verification Insights

**Generated at:** 2026-06-02 00:45:52

## Key Observations

- Verified 13 Silver Parquet artifacts.
- Grand Prix races: 55; Sprint races: 15.
- Grid-to-finish correlation: 0.742.

## Issues

- session_result.position: 149 nulls

## Recommendations

- Use event_type rather than session_type to distinguish Grand Prix races from Sprint races.
- Treat remaining critical null reviews as feature-level decisions, not file integrity failures.
- Proceed to Gold only after this Silver verification status remains PASS.

# Null Analysis Insights

**Generated at:** 2026-06-01 18:00:32

## Summary

Computed null profile for 15 endpoints and 165 endpoint-columns.

## Key Observations

- Columns with nulls: 34
- Review-level semantic nulls: 7

## Issues

- None

## Recommendations

- Treat technical key nulls as P0 only.
- Document domain semantic nulls as REVIEW for Silver strategy.
- Keep structural optional nulls out of P0 gating.

## Next Steps

- Run 05_range_validation.ipynb

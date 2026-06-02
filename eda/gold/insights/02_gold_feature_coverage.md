# Gold Feature Coverage

**Generated at:** 2026-06-02 14:28:33

## Key Observations

- Master lap features contain 63,676 rows and 58 columns.
- All generated lap-grain feature artifacts preserve canonical driver-session-lap grain.
- Sparse stint rows are retained rather than dropped, protecting label alignment.

## Issues

- None

## Recommendations

- Treat high-null feature columns explicitly during model preprocessing.
- Investigate missing stint coverage before using tyre features as mandatory model inputs.

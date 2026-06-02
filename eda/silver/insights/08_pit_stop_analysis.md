# Silver Pit Stop Analysis Insights

**Generated at:** 2026-06-02 02:13:45

## Key Observations

- Analyzed 1,891 pit records from silver.
- Median valid pit duration: 23.28s.
- Fastest team by median duration: RB.

## Issues

- None

## Recommendations

- Keep pit as a required Silver artifact; never fallback to raw files inside Silver EDA.
- Use first_pit_lap, pit_count, and team rolling pit duration as Gold strategy features.
- Treat wet sessions separately because pit timing and tyre calls change under rain risk.

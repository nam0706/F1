# Validation Report

| file                   | severity   | check          | message                                                                                         |
|:-----------------------|:-----------|:---------------|:------------------------------------------------------------------------------------------------|
| session_result.parquet | error      | missing_values | Column position has 149 nulls (10.8%)                                                           |
| car_data.parquet       | warning    | duplicate_key  | Found 43020195 duplicate rows (99.8691%) for key=['session_key', 'driver_number', 'lap_number'] |
| location.parquet       | warning    | duplicate_key  | Found 21768542 duplicate rows (99.7416%) for key=['session_key', 'driver_number', 'lap_number'] |

from __future__ import annotations

import pandas as pd


def build_telemetry_lap_features(car_data: pd.DataFrame, location: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build optional telemetry lap aggregates."""
    raise NotImplementedError("Gold telemetry features are not implemented yet.")


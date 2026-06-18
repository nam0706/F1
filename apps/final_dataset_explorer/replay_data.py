from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
LOCATION_PATH = PROJECT_ROOT / "data" / "cleaned" / "location.parquet"
CAR_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "car_data.parquet"


def available_location_sessions() -> set[int]:
    if not LOCATION_PATH.exists():
        return set()
    dataset = ds.dataset(LOCATION_PATH, format="parquet")
    table = dataset.to_table(columns=["session_key"])
    sessions = table.to_pandas()["session_key"].dropna().unique().tolist()
    return {int(session) for session in sessions}


def build_replay_payload(
    session_key: int,
    selected_drivers: tuple[int, ...],
    driver_labels: dict[int, str],
    frame_step_seconds: int = 2,
    max_frames: int = 900,
) -> dict[str, Any]:
    """Build compact replay frames from cleaned FastF1 location parquet."""
    if not LOCATION_PATH.exists() or not selected_drivers:
        return empty_payload(session_key, selected_drivers, driver_labels)

    columns = ["session_key", "driver_number", "SessionTime", "X", "Y", "lap_number", "compound", "Status"]
    dataset = ds.dataset(LOCATION_PATH, format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("session_key") == int(session_key),
    )
    session_loc = table.to_pandas()
    if session_loc.empty:
        return empty_payload(session_key, selected_drivers, driver_labels)

    track_points, bounds = build_track_cloud(session_loc)
    track_line = build_track_line(session_key)

    loc = session_loc.copy()
    loc = loc.dropna(subset=["driver_number", "SessionTime", "X", "Y"]).copy()
    loc["driver_number"] = pd.to_numeric(loc["driver_number"], errors="coerce").astype("Int64")
    loc["session_seconds"] = pd.to_timedelta(loc["SessionTime"], errors="coerce").dt.total_seconds()
    loc = loc.dropna(subset=["driver_number", "session_seconds"])
    loc["driver_number"] = loc["driver_number"].astype(int)
    loc = loc[loc["driver_number"].isin(selected_drivers)].copy()
    if loc.empty:
        return empty_payload(session_key, selected_drivers, driver_labels)

    loc["frame_second"] = (loc["session_seconds"] // frame_step_seconds * frame_step_seconds).astype(int)
    leaderboard_source = load_leaderboard_source(session_key, selected_drivers, frame_step_seconds)
    replay = (
        loc.groupby(["frame_second", "driver_number"], as_index=False)
        .agg(
            x=("X", "median"),
            y=("Y", "median"),
            lap=("lap_number", "max"),
            compound=("compound", "last"),
            status=("Status", "last"),
        )
        .sort_values(["frame_second", "driver_number"])
    )
    if replay.empty:
        return empty_payload(session_key, selected_drivers, driver_labels)

    frame_seconds = sorted(replay["frame_second"].unique().tolist())
    if max_frames and len(frame_seconds) > max_frames:
        stride = max(1, len(frame_seconds) // max_frames)
        keep = set(frame_seconds[::stride][:max_frames])
        replay = replay[replay["frame_second"].isin(keep)].copy()
        frame_seconds = sorted(replay["frame_second"].unique().tolist())

    min_t = frame_seconds[0]
    replay["t"] = replay["frame_second"] - min_t
    if track_line:
        line_bounds = pd.DataFrame(track_line)
        bounds = {
            "min_x": float(line_bounds["x"].min()),
            "max_x": float(line_bounds["x"].max()),
            "min_y": float(line_bounds["y"].min()),
            "max_y": float(line_bounds["y"].max()),
        }
    elif not track_points:
        bounds = {
            "min_x": float(replay["x"].min()),
            "max_x": float(replay["x"].max()),
            "min_y": float(replay["y"].min()),
            "max_y": float(replay["y"].max()),
        }
        unique_track = replay[["x", "y"]].dropna().drop_duplicates()
        track_points = (
            unique_track.sample(min(len(unique_track), 6000), random_state=42)
            .round(1)
            .to_dict("records")
        )

    frames = []
    for frame_second, frame_df in replay.groupby("frame_second", sort=True):
        cars = []
        leaderboard = build_leaderboard(frame_df, leaderboard_source, frame_second, driver_labels)
        relative_t = int(frame_second - min_t)
        for row in frame_df.itertuples(index=False):
            driver_number = int(row.driver_number)
            cars.append(
                {
                    "d": str(driver_number),
                    "label": driver_labels.get(driver_number, f"#{driver_number}"),
                    "x": round(float(row.x), 1),
                    "y": round(float(row.y), 1),
                    "lap": None if pd.isna(row.lap) else int(row.lap),
                    "compound": None if pd.isna(row.compound) else str(row.compound),
                    "status": None if pd.isna(row.status) else str(row.status),
                    "position": leaderboard.get(driver_number, {}).get("position"),
                }
            )
        frames.append(
            {
                "t": relative_t,
                "cars": cars,
                "leaderboard": list(leaderboard.values()),
            }
        )

    drivers = [
        {"driver_number": str(driver), "label": driver_labels.get(driver, f"#{driver}")}
        for driver in selected_drivers
    ]
    return {
        "session_key": int(session_key),
        "frame_step_seconds": int(frame_step_seconds),
        "drivers": drivers,
        "bounds": bounds,
        "track": track_points,
        "track_line": track_line,
        "frames": frames,
    }


def empty_payload(session_key: int, selected_drivers: tuple[int, ...], driver_labels: dict[int, str]) -> dict[str, Any]:
    return {
        "session_key": int(session_key),
        "frame_step_seconds": 0,
        "drivers": [
            {"driver_number": str(driver), "label": driver_labels.get(driver, f"#{driver}")}
            for driver in selected_drivers
        ],
        "bounds": {},
        "track": [],
        "track_line": [],
        "frames": [],
    }


def build_track_cloud(session_loc: pd.DataFrame, max_points: int = 9000) -> tuple[list[dict[str, float]], dict[str, float]]:
    track = session_loc.dropna(subset=["X", "Y"])[["X", "Y"]].copy()
    if track.empty:
        return [], {}

    x_low, x_high = track["X"].quantile([0.005, 0.995])
    y_low, y_high = track["Y"].quantile([0.005, 0.995])
    track = track[
        track["X"].between(x_low, x_high)
        & track["Y"].between(y_low, y_high)
    ].copy()
    if track.empty:
        return [], {}

    track["x_bin"] = (track["X"] / 12).round().astype("int64")
    track["y_bin"] = (track["Y"] / 12).round().astype("int64")
    cloud = (
        track.groupby(["x_bin", "y_bin"], as_index=False)
        .agg(x=("X", "median"), y=("Y", "median"), samples=("X", "size"))
        .sort_values("samples", ascending=False)
    )
    if len(cloud) > max_points:
        cloud = cloud.head(max_points)

    bounds = {
        "min_x": float(cloud["x"].min()),
        "max_x": float(cloud["x"].max()),
        "min_y": float(cloud["y"].min()),
        "max_y": float(cloud["y"].max()),
    }
    points = cloud[["x", "y"]].round(1).to_dict("records")
    return points, bounds


def build_track_line(session_key: int, bins: int = 520) -> list[dict[str, float]]:
    if not CAR_DATA_PATH.exists():
        return []

    dataset = ds.dataset(CAR_DATA_PATH, format="parquet")
    available = set(dataset.schema.names)
    columns = [
        col
        for col in ["session_key", "driver_number", "X", "Y", "Distance", "RelativeDistance", "Speed", "lap_number"]
        if col in available
    ]
    if not {"X", "Y"}.issubset(columns):
        return []

    table = dataset.to_table(columns=columns, filter=ds.field("session_key") == int(session_key))
    car = table.to_pandas()
    if car.empty:
        return []

    car = car.dropna(subset=["X", "Y"]).copy()
    for col in ["Distance", "RelativeDistance", "Speed", "lap_number"]:
        if col in car.columns:
            car[col] = pd.to_numeric(car[col], errors="coerce")

    if "RelativeDistance" in car.columns and car["RelativeDistance"].notna().sum() > 100:
        car = car.dropna(subset=["RelativeDistance"]).copy()
        car["progress"] = car["RelativeDistance"].clip(0, 1)
    elif {"Distance", "driver_number", "lap_number"}.issubset(car.columns):
        car = car.dropna(subset=["Distance", "driver_number", "lap_number"]).copy()
        car["lap_min"] = car.groupby(["driver_number", "lap_number"])["Distance"].transform("min")
        car["lap_max"] = car.groupby(["driver_number", "lap_number"])["Distance"].transform("max")
        span = (car["lap_max"] - car["lap_min"]).replace(0, pd.NA)
        car["progress"] = ((car["Distance"] - car["lap_min"]) / span).clip(0, 1)
    else:
        return []

    car = car.dropna(subset=["progress", "X", "Y"])
    if "Speed" in car.columns:
        speed = pd.to_numeric(car["Speed"], errors="coerce")
        car = car[(speed.isna()) | (speed > 20)].copy()
    if car.empty:
        return []

    car["progress_bin"] = (car["progress"] * (bins - 1)).round().astype(int)
    line = (
        car.groupby("progress_bin", as_index=False)
        .agg(
            x=("X", "median"),
            y=("Y", "median"),
            progress=("progress", "median"),
            samples=("X", "size"),
        )
        .sort_values("progress")
    )
    if line.empty:
        return []

    min_samples = max(2, int(car["driver_number"].nunique() * 0.1)) if "driver_number" in car.columns else 2
    line = line[line["samples"] >= min_samples].copy()
    line = remove_line_outliers(line)
    line = smooth_track_line(line)
    if len(line) < 30:
        return []
    line["sector"] = line["progress"].apply(progress_to_sector)
    return line[["x", "y", "sector"]].round(1).to_dict("records")


def progress_to_sector(progress: float) -> int:
    if pd.isna(progress):
        return 1
    if progress < 1 / 3:
        return 1
    if progress < 2 / 3:
        return 2
    return 3


def remove_line_outliers(line: pd.DataFrame) -> pd.DataFrame:
    if len(line) < 4:
        return line
    dx = line["x"].diff()
    dy = line["y"].diff()
    step = (dx.pow(2) + dy.pow(2)).pow(0.5)
    normal = step[step > 0]
    if normal.empty:
        return line
    threshold = max(float(normal.quantile(0.95)) * 2.5, float(normal.median()) * 8, 400.0)
    return line.loc[step.isna() | (step <= threshold)].copy()


def smooth_track_line(line: pd.DataFrame, window: int = 9) -> pd.DataFrame:
    if len(line) < window:
        return line
    work = line.sort_values("progress").reset_index(drop=True).copy()
    pad = window // 2
    wrapped = pd.concat([work.tail(pad), work, work.head(pad)], ignore_index=True)
    for col in ["x", "y"]:
        smoothed = wrapped[col].rolling(window=window, center=True, min_periods=1).median()
        work[col] = smoothed.iloc[pad : pad + len(work)].to_numpy()
    return work


def load_leaderboard_source(session_key: int, selected_drivers: tuple[int, ...], frame_step_seconds: int) -> pd.DataFrame:
    if not CAR_DATA_PATH.exists():
        return pd.DataFrame()

    dataset = ds.dataset(CAR_DATA_PATH, format="parquet")
    available = set(dataset.schema.names)
    columns = [
        col
        for col in ["session_key", "driver_number", "SessionTime", "lap_number", "Distance", "Speed", "compound", "Status"]
        if col in available
    ]
    if not {"driver_number", "SessionTime"}.issubset(columns):
        return pd.DataFrame()

    table = dataset.to_table(columns=columns, filter=ds.field("session_key") == int(session_key))
    car = table.to_pandas()
    if car.empty:
        return pd.DataFrame()

    car = car.dropna(subset=["driver_number", "SessionTime"]).copy()
    car["driver_number"] = pd.to_numeric(car["driver_number"], errors="coerce").astype("Int64")
    car["session_seconds"] = pd.to_timedelta(car["SessionTime"], errors="coerce").dt.total_seconds()
    car = car.dropna(subset=["driver_number", "session_seconds"])
    car["driver_number"] = car["driver_number"].astype(int)
    car = car[car["driver_number"].isin(selected_drivers)].copy()
    if car.empty:
        return pd.DataFrame()

    car["frame_second"] = (car["session_seconds"] // frame_step_seconds * frame_step_seconds).astype(int)
    if "Distance" not in car.columns:
        car["Distance"] = pd.NA
    if "Speed" not in car.columns:
        car["Speed"] = pd.NA
    if "lap_number" not in car.columns:
        car["lap_number"] = pd.NA
    if "compound" not in car.columns:
        car["compound"] = pd.NA
    if "Status" not in car.columns:
        car["Status"] = pd.NA

    return (
        car.groupby(["frame_second", "driver_number"], as_index=False)
        .agg(
            lap=("lap_number", "max"),
            distance=("Distance", "median"),
            speed=("Speed", "median"),
            compound=("compound", "last"),
            status=("Status", "last"),
        )
        .sort_values(["frame_second", "driver_number"])
    )


def build_leaderboard(
    frame_df: pd.DataFrame,
    leaderboard_source: pd.DataFrame,
    frame_second: int,
    driver_labels: dict[int, str],
) -> dict[int, dict[str, Any]]:
    if not leaderboard_source.empty:
        work = leaderboard_source[leaderboard_source["frame_second"] == frame_second].copy()
    else:
        work = frame_df.rename(columns={"x": "distance"}).copy()
        work["speed"] = pd.NA

    if work.empty:
        return {}

    work["lap_sort"] = pd.to_numeric(work.get("lap", pd.Series(index=work.index)), errors="coerce").fillna(-1)
    work["distance_sort"] = pd.to_numeric(work.get("distance", pd.Series(index=work.index)), errors="coerce").fillna(-1)
    work = work.sort_values(["lap_sort", "distance_sort"], ascending=[False, False]).reset_index(drop=True)

    result = {}
    leader_progress = None
    for position, row in enumerate(work.itertuples(index=False), start=1):
        driver_number = int(row.driver_number)
        lap = getattr(row, "lap", None)
        distance = getattr(row, "distance", None)
        speed = getattr(row, "speed", None)
        compound = getattr(row, "compound", None)
        progress = None if pd.isna(distance) else float(distance)
        if leader_progress is None:
            leader_progress = progress
        gap = None if progress is None or leader_progress is None else max(0.0, leader_progress - progress)
        result[driver_number] = {
            "position": position,
            "driver": str(driver_number),
            "label": driver_labels.get(driver_number, f"#{driver_number}"),
            "lap": None if pd.isna(lap) else int(lap),
            "speed": None if pd.isna(speed) else round(float(speed), 0),
            "compound": None if pd.isna(compound) else str(compound),
            "gap": None if gap is None else round(gap, 1),
        }
    return result

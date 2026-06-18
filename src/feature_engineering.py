"""
src/feature_engineering.py
Lap-level feature engineering for F1 real-time win rate prediction.
Input:  data/cleaned/*.csv
Output: data/processed/master_dataset.csv  (granularity: session_key + driver_number + lap_number)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from .config import load_config
from .utils import setup_logging, write_csv, write_json, utc_now_iso

logger = logging.getLogger(__name__)

# ── Domain-knowledge tyre cliff defaults (lap number within a stint) ──────────
TYRE_CLIFF_DEFAULTS = {
    "SOFT": 22, "MEDIUM": 33, "HARD": 45,
    "INTERMEDIATE": 20, "WET": 15, "UNKNOWN": 30,
}
TYRE_CLIFF_BOUNDS = {
    "SOFT": (12, 34),
    "MEDIUM": (18, 48),
    "HARD": (25, 60),
    "INTERMEDIATE": (10, 32),
    "WET": (10, 28),
    "UNKNOWN": (12, 45),
}
ROLLING_WINDOWS = [3, 5, 10]
WINSOR_Q = (0.01, 0.99)  # Winsorize at 1 – 99 percentile
WEATHER_ASOF_TOLERANCE = pd.Timedelta("10min")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tyre cliff learning (hybrid: data-learned + domain knowledge)
# ─────────────────────────────────────────────────────────────────────────────

def learn_tyre_cliffs(stints_df: pd.DataFrame, laps_df: pd.DataFrame) -> dict[str, int]:
    """
    Học tyre cliff từ dữ liệu thực tế.
    Cliff = lap trong stint mà lap_duration bắt đầu tăng liên tục > 1% per bin.
    Kết quả được blend 60% learned + 40% domain knowledge.
    """
    cliffs = dict(TYRE_CLIFF_DEFAULTS)
    needed_s = {"session_key", "driver_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"}
    needed_l = {"session_key", "driver_number", "lap_number", "lap_duration"}
    if stints_df.empty or laps_df.empty:
        return cliffs
    if not needed_s.issubset(stints_df.columns) or not needed_l.issubset(laps_df.columns):
        return cliffs

    merged = laps_df[list(needed_l)].merge(stints_df[list(needed_s)], on=["session_key", "driver_number"], how="left")
    merged = merged[
        (merged["lap_number"] >= merged["lap_start"].fillna(-1)) &
        (merged["lap_number"] <= merged["lap_end"].fillna(9999))
    ].copy()
    merged["tyre_age"] = (merged["lap_number"] - merged["lap_start"] + merged["tyre_age_at_start"].fillna(0))

    for compound in merged["compound"].dropna().unique():
        cdf = merged[merged["compound"] == compound].copy()
        if len(cdf) < 50:
            continue
        try:
            cdf["age_bin"] = pd.cut(cdf["tyre_age"], bins=range(0, 62, 2), right=False)
            perf = cdf.groupby("age_bin", observed=True)["lap_duration"].median().dropna()
            if len(perf) < 5:
                continue
            smoothed = perf.rolling(3, min_periods=2, center=True).median()
            pct_change = smoothed.pct_change()
            compound_key = str(compound).upper()
            domain = TYRE_CLIFF_DEFAULTS.get(compound_key, TYRE_CLIFF_DEFAULTS["UNKNOWN"])
            lower, upper = TYRE_CLIFF_BOUNDS.get(compound_key, TYRE_CLIFF_BOUNDS["UNKNOWN"])
            candidates = []
            for interval, pct in pct_change.items():
                if pd.isna(pct) or pct < 0.018:
                    continue
                learned_age = int(interval.left)
                if learned_age < lower:
                    continue
                next_pos = pct_change.index.get_loc(interval) + 1
                if next_pos < len(pct_change) and pct_change.iloc[next_pos] < -0.01:
                    continue
                candidates.append(learned_age)
            if not candidates:
                continue
            learned = int(np.clip(candidates[0], lower, upper))
            blended = int(np.clip(round(0.4 * learned + 0.6 * domain), lower, upper))
            cliffs[compound_key] = blended
            logger.info(f"  Tyre cliff [{compound}]: learned={learned} domain={domain} → blended={blended}")
        except Exception as exc:
            logger.warning(f"  Cliff learning failed for {compound}: {exc}")
    return cliffs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stint features per lap
# ─────────────────────────────────────────────────────────────────────────────

def add_stint_features(laps: pd.DataFrame, stints: pd.DataFrame, cliffs: dict) -> pd.DataFrame:
    """Merge stints → compound, tyre age, cliff features per lap."""
    if stints.empty:
        laps["compound"] = "UNKNOWN"
        laps["current_tyre_age"] = np.nan
        laps["stint_number"] = np.nan
        return laps

    needed = {"session_key", "driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"}
    if not needed.issubset(stints.columns):
        return laps

    lap_keys = ["session_key", "driver_number", "lap_number"]
    merged = laps.merge(stints[list(needed)], on=["session_key", "driver_number"], how="left")
    in_range = (
        (merged["lap_number"] >= merged["lap_start"].fillna(-1)) &
        (merged["lap_number"] <= merged["lap_end"].fillna(9999))
    )
    matched = merged[in_range].copy()
    if matched.empty:
        out = laps.copy()
        out["compound"] = "UNKNOWN"
        out["current_tyre_age"] = np.nan
        out["stint_number"] = np.nan
        out["compound_cliff"] = TYRE_CLIFF_DEFAULTS["UNKNOWN"]
        out["laps_until_cliff"] = np.nan
        out["is_past_cliff"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["has_stint_match"] = False
        return out

    matched = (matched.sort_values("stint_number")
               .groupby(lap_keys, as_index=False)
               .first())

    matched["current_tyre_age"] = (matched["lap_number"] - matched["lap_start"] +
                                    matched["tyre_age_at_start"].fillna(0)).clip(lower=0)

    compound_upper = matched["compound"].fillna("UNKNOWN").str.upper()
    matched["compound"] = compound_upper
    matched["compound_cliff"] = compound_upper.map(cliffs).fillna(TYRE_CLIFF_DEFAULTS["UNKNOWN"])
    matched["laps_until_cliff"] = (matched["compound_cliff"] - matched["current_tyre_age"]).clip(lower=0)
    matched["is_past_cliff"] = (matched["current_tyre_age"] > matched["compound_cliff"]).astype("Int64")
    matched["has_stint_match"] = True

    stint_cols = [
        "stint_number",
        "compound",
        "current_tyre_age",
        "compound_cliff",
        "laps_until_cliff",
        "is_past_cliff",
        "has_stint_match",
    ]
    out = laps.merge(matched[lap_keys + stint_cols], on=lap_keys, how="left")
    out["compound"] = out["compound"].fillna("UNKNOWN")
    out["compound_cliff"] = out["compound_cliff"].fillna(TYRE_CLIFF_DEFAULTS["UNKNOWN"])
    out["has_stint_match"] = out["has_stint_match"].eq(True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rolling window features (no future leak — shift(1) before rolling)
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["session_key", "driver_number", "lap_number"])
    grp = df.groupby(["session_key", "driver_number"])["lap_duration"]
    for w in ROLLING_WINDOWS:
        shifted = grp.transform(lambda x: x.shift(1))
        df[f"rolling_avg_lap_{w}"] = shifted.groupby(
            [df["session_key"], df["driver_number"]]
        ).transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"rolling_std_lap_{w}"] = shifted.groupby(
            [df["session_key"], df["driver_number"]]
        ).transform(lambda x: x.rolling(w, min_periods=2).std())
        df[f"rolling_best_lap_{w}"] = shifted.groupby(
            [df["session_key"], df["driver_number"]]
        ).transform(lambda x: x.rolling(w, min_periods=1).min())

    df["lap_delta_prev"] = df.groupby(["session_key", "driver_number"])["lap_duration"].diff()
    total_laps = df.groupby(["session_key", "driver_number"])["lap_number"].transform("max")
    df["laps_to_go"] = (total_laps - df["lap_number"]).clip(lower=0)
    df["race_completion_pct"] = (df["lap_number"] / total_laps.replace(0, np.nan)).round(4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Weather features + Track Evolution
# ─────────────────────────────────────────────────────────────────────────────

def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty or "date_start" not in df.columns or "date" not in weather.columns:
        return df

    df = df.copy()
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce", utc=True)

    w_cols = [c for c in ["session_key", "date", "air_temperature", "track_temperature",
                           "humidity", "rainfall", "wind_speed"] if c in weather.columns]
    weather_sub = weather[w_cols].dropna(subset=["date"]).sort_values("date")

    parts = []
    for sk, lap_grp in df.groupby("session_key"):
        wg = weather_sub[weather_sub["session_key"] == sk].copy()
        if wg.empty:
            parts.append(lap_grp)
            continue
        merged = pd.merge_asof(
            lap_grp.sort_values("date_start"),
            wg.drop(columns=["session_key"]),
            left_on="date_start",
            right_on="date",
            direction="backward",
            tolerance=WEATHER_ASOF_TOLERANCE,
        ).drop(columns=["date"], errors="ignore")

        if "track_temperature" in merged.columns:
            base_temp = wg["track_temperature"].iloc[0]
            merged["track_temp_evolution"] = (merged["track_temperature"] - base_temp).round(2)
        parts.append(merged)

    return pd.concat(parts, ignore_index=True) if parts else df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Position features per lap
# ─────────────────────────────────────────────────────────────────────────────

def add_position_features(df: pd.DataFrame, position: pd.DataFrame) -> pd.DataFrame:
    """Legacy OpenF1 position endpoint features; not used by the live-before-lap build."""
    if position.empty or "date_start" not in df.columns:
        return df
    required = {"session_key", "driver_number", "date", "position"}
    if not required.issubset(position.columns):
        return df

    df = df.copy()
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
    position = position.copy()
    position["date"] = pd.to_datetime(position["date"], errors="coerce", utc=True)

    parts = []
    for (sk, dn), grp in df.groupby(["session_key", "driver_number"]):
        pg = position[(position["session_key"] == sk) & (position["driver_number"] == dn)][
            ["date", "position"]].dropna().sort_values("date")
        if pg.empty:
            parts.append(grp)
            continue
        merged = pd.merge_asof(
            grp.sort_values("date_start"), pg,
            left_on="date_start", right_on="date", direction="backward"
        ).drop(columns=["date"], errors="ignore").rename(columns={"position": "current_position"})
        parts.append(merged)

    if not parts:
        return df
    out = pd.concat(parts, ignore_index=True)
    if "current_position" in out.columns:
        cp = pd.to_numeric(out["current_position"], errors="coerce")
        out["is_leading"] = (cp == 1).astype("Int64")
        out["is_top3"] = cp.between(1, 3).astype("Int64")
        out["is_top10"] = cp.between(1, 10).astype("Int64")
        out["position_gain_lap"] = out.groupby(["session_key", "driver_number"])["current_position"].diff().mul(-1)
    return out


def add_pit_features(df: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    if pit.empty:
        return df
    required = {"session_key", "driver_number", "lap_number"}
    if not required.issubset(pit.columns):
        return df

    pit_df = pit.copy()
    for col in ["session_key", "driver_number", "lap_number", "pit_duration", "stop_duration", "lane_duration"]:
        if col in pit_df.columns:
            pit_df[col] = pd.to_numeric(pit_df[col], errors="coerce")
    pit_df = pit_df.dropna(subset=list(required))
    if pit_df.empty:
        return df

    agg_spec = {"pit_stop_count_lap": ("lap_number", "size")}
    if "pit_duration" in pit_df.columns:
        agg_spec["pit_duration_sum_lap"] = ("pit_duration", "sum")
        agg_spec["pit_duration_mean_lap"] = ("pit_duration", "mean")
    if "stop_duration" in pit_df.columns:
        agg_spec["stop_duration_mean_lap"] = ("stop_duration", "mean")
    if "lane_duration" in pit_df.columns:
        agg_spec["lane_duration_mean_lap"] = ("lane_duration", "mean")

    pit_lap = (
        pit_df.groupby(["session_key", "driver_number", "lap_number"], as_index=False)
        .agg(**agg_spec)
    )
    out = df.merge(pit_lap, on=["session_key", "driver_number", "lap_number"], how="left")

    pit_cols = [c for c in pit_lap.columns if c not in {"session_key", "driver_number", "lap_number"}]
    for col in pit_cols:
        out[col] = out[col].fillna(0)

    out = out.sort_values(["session_key", "driver_number", "lap_number"])
    grp = out.groupby(["session_key", "driver_number"])
    out["pit_stop_count_before_lap"] = grp["pit_stop_count_lap"].cumsum() - out["pit_stop_count_lap"]
    out["pit_duration_sum_before_lap"] = (
        grp["pit_duration_sum_lap"].cumsum() - out["pit_duration_sum_lap"]
        if "pit_duration_sum_lap" in out.columns else 0
    )
    out["pit_stop_count_so_far"] = out["pit_stop_count_before_lap"]
    out["is_pit_lap"] = (out["pit_stop_count_lap"] > 0).astype("Int64")
    out["was_pit_previous_lap"] = grp["is_pit_lap"].shift(1).fillna(0).astype("Int64")
    pit_lap_number = out["lap_number"].where(out["is_pit_lap"].astype(bool))
    out["_last_pit_lap"] = pit_lap_number.groupby([out["session_key"], out["driver_number"]]).ffill().groupby(
        [out["session_key"], out["driver_number"]]
    ).shift(1)
    out["laps_since_last_pit"] = out["lap_number"] - out["_last_pit_lap"]
    out.drop(columns=["_last_pit_lap"], inplace=True)
    out = out.rename(columns={
        "pit_stop_count_lap": "pit_stop_count_current_lap",
        "pit_duration_sum_lap": "pit_duration_sum_current_lap",
        "pit_duration_mean_lap": "pit_duration_mean_current_lap",
        "stop_duration_mean_lap": "stop_duration_mean_current_lap",
        "lane_duration_mean_lap": "lane_duration_mean_current_lap",
        "is_pit_lap": "is_pit_current_lap",
    })
    return out


def add_race_control_features(df: pd.DataFrame, race_control: pd.DataFrame) -> pd.DataFrame:
    if race_control.empty or "session_key" not in race_control.columns:
        return df

    rc = race_control.copy()
    lap_col = "lap_number_clean" if "lap_number_clean" in rc.columns else "lap_number"
    if lap_col not in rc.columns:
        return df
    rc["event_lap_number"] = pd.to_numeric(rc[lap_col], errors="coerce")
    rc = rc.dropna(subset=["session_key", "event_lap_number"])
    if rc.empty:
        return df

    flag_series = rc["flag"].fillna("").astype(str).str.upper() if "flag" in rc.columns else pd.Series("", index=rc.index)
    rc["is_yellow_flag_event"] = flag_series.str.contains("YELLOW|SAFETY|VSC|RED", regex=True)
    rc["is_green_flag_event"] = flag_series.str.contains("GREEN", regex=True)
    rc_lap = (
        rc.groupby(["session_key", "event_lap_number"], as_index=False)
        .agg(
            race_control_events_lap=("session_key", "size"),
            yellow_flag_events_lap=("is_yellow_flag_event", "sum"),
            green_flag_events_lap=("is_green_flag_event", "sum"),
        )
        .rename(columns={"event_lap_number": "lap_number"})
    )
    out = df.merge(rc_lap, on=["session_key", "lap_number"], how="left")
    for col in ["race_control_events_lap", "yellow_flag_events_lap", "green_flag_events_lap"]:
        out[col] = out[col].fillna(0)

    out = out.sort_values(["session_key", "driver_number", "lap_number"])
    grp = out.groupby(["session_key", "driver_number"])
    out["yellow_flag_active_lap"] = (out["yellow_flag_events_lap"] > 0).astype("Int64")
    out["race_control_events_before_lap"] = grp["race_control_events_lap"].cumsum() - out["race_control_events_lap"]
    out["yellow_flag_events_before_lap"] = grp["yellow_flag_events_lap"].cumsum() - out["yellow_flag_events_lap"]
    out["green_flag_events_before_lap"] = grp["green_flag_events_lap"].cumsum() - out["green_flag_events_lap"]
    out["race_control_events_so_far"] = out["race_control_events_before_lap"]
    out["yellow_flag_previous_lap"] = grp["yellow_flag_active_lap"].shift(1).fillna(0).astype("Int64")
    out = out.rename(columns={
        "race_control_events_lap": "race_control_events_current_lap",
        "yellow_flag_events_lap": "yellow_flag_events_current_lap",
        "green_flag_events_lap": "green_flag_events_current_lap",
        "yellow_flag_active_lap": "yellow_flag_active_current_lap",
    })
    return out


def add_overtake_features(df: pd.DataFrame, overtakes: pd.DataFrame) -> pd.DataFrame:
    if overtakes.empty or "session_key" not in overtakes.columns:
        return df
    if "date_start" not in df.columns or "date" not in overtakes.columns:
        return df

    ot = overtakes.copy()
    if "overtaking_driver_number" not in ot.columns or "overtaken_driver_number" not in ot.columns:
        return df
    ot["date"] = pd.to_datetime(ot["date"], errors="coerce", utc=True)
    for col in ["session_key", "overtaking_driver_number", "overtaken_driver_number"]:
        ot[col] = pd.to_numeric(ot[col], errors="coerce")
    ot = ot.dropna(subset=["session_key", "overtaking_driver_number", "overtaken_driver_number", "date"])
    if ot.empty:
        return df

    made = ot.rename(columns={"overtaking_driver_number": "driver_number"})
    made = made.groupby(["session_key", "driver_number", "date"], as_index=False).size()
    made = made.rename(columns={"size": "overtakes_made_event"})
    made = made.sort_values(["session_key", "driver_number", "date"])
    made["overtakes_made_so_far"] = made.groupby(["session_key", "driver_number"])["overtakes_made_event"].cumsum()

    lost = ot.rename(columns={"overtaken_driver_number": "driver_number"})
    lost = lost.groupby(["session_key", "driver_number", "date"], as_index=False).size()
    lost = lost.rename(columns={"size": "overtakes_lost_event"})
    lost = lost.sort_values(["session_key", "driver_number", "date"])
    lost["overtakes_lost_so_far"] = lost.groupby(["session_key", "driver_number"])["overtakes_lost_event"].cumsum()

    out = df.copy()
    out["date_start"] = pd.to_datetime(out["date_start"], errors="coerce", utc=True)
    parts = []
    for (sk, dn), grp in out.groupby(["session_key", "driver_number"]):
        grp = grp.sort_values("date_start")
        made_g = made[(made["session_key"] == sk) & (made["driver_number"] == dn)][["date", "overtakes_made_so_far"]]
        lost_g = lost[(lost["session_key"] == sk) & (lost["driver_number"] == dn)][["date", "overtakes_lost_so_far"]]

        merged = grp
        if not made_g.empty:
            merged = pd.merge_asof(
                merged.sort_values("date_start"),
                made_g.sort_values("date"),
                left_on="date_start",
                right_on="date",
                direction="backward",
            ).drop(columns=["date"], errors="ignore")
        if not lost_g.empty:
            merged = pd.merge_asof(
                merged.sort_values("date_start"),
                lost_g.sort_values("date"),
                left_on="date_start",
                right_on="date",
                direction="backward",
            ).drop(columns=["date"], errors="ignore")
        parts.append(merged)

    out = pd.concat(parts, ignore_index=True) if parts else df
    if "overtakes_made_so_far" not in out.columns:
        out["overtakes_made_so_far"] = 0
    if "overtakes_lost_so_far" not in out.columns:
        out["overtakes_lost_so_far"] = 0
    out["overtakes_made_so_far"] = out["overtakes_made_so_far"].fillna(0)
    out["overtakes_lost_so_far"] = out["overtakes_lost_so_far"].fillna(0)
    out["net_overtakes_so_far"] = out["overtakes_made_so_far"] - out["overtakes_lost_so_far"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. ML Targets
# ─────────────────────────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        for col in ["final_position", "target_win", "target_podium", "target_top10"]:
            df[col] = pd.NA
        return df

    res_cols = [c for c in ["session_key", "driver_number", "position", "status"] if c in results.columns]
    res = results[res_cols].rename(columns={"position": "final_position"})
    df = df.merge(res, on=["session_key", "driver_number"], how="left")
    fp = pd.to_numeric(df["final_position"], errors="coerce")
    df["target_win"] = (fp == 1).astype("Int64")
    df["target_podium"] = fp.between(1, 3).astype("Int64")
    df["target_top10"] = fp.between(1, 10).astype("Int64")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Winsorization
# ─────────────────────────────────────────────────────────────────────────────

def winsorize(df: pd.DataFrame) -> pd.DataFrame:
    skip = {
        "session_key", "driver_number", "lap_number", "circuit_key", "year",
        "stint_number", "compound_cliff", "final_position",
    }
    skip_prefixes = (
        "target_", "is_", "has_", "was_", "rolling_", "laps_",
        "pit_stop_count", "race_control_events", "yellow_flag_events",
        "green_flag_events", "overtakes_", "net_overtakes_",
    )
    skip_suffixes = ("_count", "_events_lap", "_before_lap", "_so_far", "_lap")
    for col in df.select_dtypes(include="number").columns:
        if (
            col in skip
            or any(col.startswith(p) for p in skip_prefixes)
            or any(col.endswith(s) for s in skip_suffixes)
        ):
            continue
        lo, hi = df[col].quantile(WINSOR_Q[0]), df[col].quantile(WINSOR_Q[1])
        if pd.isna(lo) or pd.isna(hi):
            continue
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Telemetry Aggregation: Mili-giây → Cấp độ Vòng (Gold Layer)
# ─────────────────────────────────────────────────────────────────────────────

def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return leakage-safe numeric inputs for the Live before lap N contract."""
    blocked_exact = {
        "session_key", "driver_number", "lap_number", "year", "circuit_key",
        "final_position", "status",
        "pit_stop_count_current_lap", "pit_duration_sum_current_lap",
        "pit_duration_mean_current_lap", "stop_duration_mean_current_lap",
        "lane_duration_mean_current_lap", "is_pit_current_lap",
        "race_control_events_current_lap", "yellow_flag_events_current_lap",
        "green_flag_events_current_lap", "yellow_flag_active_current_lap",
    }
    blocked_prefixes = ("target_",)
    numeric_cols = df.select_dtypes(include="number").columns
    return [
        col for col in numeric_cols
        if col not in blocked_exact and not any(col.startswith(prefix) for prefix in blocked_prefixes)
    ]


def move_outcomes_to_end(df: pd.DataFrame) -> pd.DataFrame:
    """Keep features first and labels/outcomes at the end of the dataset."""
    outcome_cols = [col for col in ["status", "final_position", "target_win", "target_podium", "target_top10"] if col in df.columns]
    feature_cols = [col for col in df.columns if col not in outcome_cols]
    return df[feature_cols + outcome_cols]


def aggregate_telemetry_to_lap(car_data: pd.DataFrame) -> pd.DataFrame:
    """
    Nén dữ liệu telemetry từ cấp độ mili-giây về cấp độ vòng chạy.

    Kỹ thuật "Nén đa chiều" — giữ lại hình dáng của chuỗi thời gian:
      - max(Speed): Tốc độ tối đa trên đoạn thẳng (sức mạnh động cơ / DRS)
      - min(Speed): Tốc độ vào cua (khả năng kiểm soát góc cua)
      - mean(Throttle): Tỷ lệ đạp ga trung bình
      - q90(Throttle): Giữ trạng thái gần lút ga trên đoạn thẳng
      - std(RPM): Độ nhất quán khi đi số (số gật cục = std cao)

    Chống Data Leakage:
      Vòng 10 chỉ được nhìn thấy telemetry của Vòng 9 (dung shift(1)).

    Args:
        car_data: DataFrame từ cleaned/car_data.parquet.

    Returns:
        DataFrame ở cấp độ (session_key, driver_number, lap_number)
        với các cột telemetry đã được dịch sang vòng trước (lag-1).
    """
    required = {"session_key", "driver_number", "lap_number"}
    if car_data.empty or not required.issubset(car_data.columns):
        return pd.DataFrame()

    group_keys = ["session_key", "driver_number", "lap_number"]

    # Xây dựng các hàm aggregation theo từng cột có trong data
    agg_spec = {}
    if "Speed" in car_data.columns:
        agg_spec["Speed"] = ["max", "min", "mean", "std"]
    if "Throttle" in car_data.columns:
        agg_spec["Throttle"] = ["mean", lambda x: x.quantile(0.9)]
    if "Brake" in car_data.columns:
        agg_spec["Brake"] = ["mean", "sum"]      # mean=tỷ lệ phanh, sum=tổng lần phanh
    if "RPM" in car_data.columns:
        agg_spec["RPM"] = ["mean", "std"]
    if "nGear" in car_data.columns:
        agg_spec["nGear"] = ["mean", "max"]
    if "DRS" in car_data.columns:
        agg_spec["DRS"] = ["sum"]               # tổng số điểm dữ liệu khi DRS bật

    if not agg_spec:
        return pd.DataFrame()

    agg = car_data.groupby(group_keys).agg(agg_spec)
    agg.columns = ["_".join([col, fn if not callable(fn) else "q90"]).strip()
                   for col, fn in agg.columns]
    agg = agg.reset_index()
    agg.columns = [c.lower() for c in agg.columns]   # Chuẩn hóa tên cột lowercase

    # ── Chống Data Leakage: shift(1) — Vòng N chỉ nhìn thấy dữ liệu Vòng N-1 ──
    agg = agg.sort_values(["session_key", "driver_number", "lap_number"])
    tele_feature_cols = [c for c in agg.columns if c not in group_keys]
    agg[tele_feature_cols] = agg.groupby(["session_key", "driver_number"])[tele_feature_cols].shift(1)

    # Đổi tên để rõ ràng là dữ liệu từ vòng trước (tele_prev_*)
    rename_map = {col: f"tele_prev_{col}" for col in tele_feature_cols}
    agg = agg.rename(columns=rename_map)

    logger.info(f"Telemetry aggregated: {len(agg):,} lap-level rows | {len(tele_feature_cols)} features (lag-1)")
    return agg


def _combine_count_sum_sumsq(stats: pd.DataFrame, prefix: str) -> pd.DataFrame:
    count_col = f"{prefix}_count"
    sum_col = f"{prefix}_sum"
    sumsq_col = f"{prefix}_sumsq"
    out = pd.DataFrame(index=stats.index)

    if count_col not in stats.columns or sum_col not in stats.columns:
        return out

    counts = stats[count_col].replace(0, np.nan)
    out[f"{prefix}_mean"] = stats[sum_col] / counts

    if sumsq_col in stats.columns:
        numerator = stats[sumsq_col] - (stats[sum_col] ** 2 / counts)
        variance = numerator / (counts - 1)
        out[f"{prefix}_std"] = np.sqrt(variance.clip(lower=0)).where(counts > 1)

    return out


def _throttle_q90_from_counts(stats: pd.DataFrame) -> pd.Series:
    throttle_cols = [c for c in stats.columns if c.startswith("throttle_bin_")]
    if not throttle_cols:
        return pd.Series(index=stats.index, dtype="float64")

    throttle_cols = sorted(throttle_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    counts = stats[throttle_cols].to_numpy(dtype="float64")
    totals = counts.sum(axis=1)
    thresholds = np.ceil(totals * 0.9)
    cumulative = np.cumsum(counts, axis=1)
    bins = np.array([int(c.rsplit("_", 1)[1]) for c in throttle_cols], dtype="float64")
    q90_idx = (cumulative >= thresholds[:, None]).argmax(axis=1)
    q90 = bins[q90_idx]
    q90[totals == 0] = np.nan
    return pd.Series(q90, index=stats.index)


def aggregate_telemetry_parquet_to_lap(path: Path, batch_size: int = 1_000_000) -> pd.DataFrame:
    """
    Aggregate cleaned/car_data.parquet without loading the full telemetry table.

    The resulting feature contract matches aggregate_telemetry_to_lap():
    lap-level telemetry features are shifted by one lap to prevent leakage.
    """
    if not path.exists():
        return pd.DataFrame()

    pf = pq.ParquetFile(path)
    schema_cols = set(pf.schema_arrow.names)
    group_keys = ["session_key", "driver_number", "lap_number"]
    if not set(group_keys).issubset(schema_cols):
        return pd.DataFrame()

    telemetry_cols = [c for c in ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"] if c in schema_cols]
    if not telemetry_cols:
        return pd.DataFrame()

    columns = group_keys + telemetry_cols
    partials = []
    logger.info(
        "Telemetry parquet aggregation: %s rows, %s row groups, selected columns=%s",
        f"{pf.metadata.num_rows:,}",
        pf.metadata.num_row_groups,
        telemetry_cols,
    )

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        chunk = batch.to_pandas()
        if chunk.empty:
            continue

        for col in columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=group_keys)
        if chunk.empty:
            continue

        grouped = chunk.groupby(group_keys, dropna=False)
        parts = []

        if "Speed" in chunk.columns:
            speed = grouped["Speed"].agg(["count", "sum", "min", "max"])
            speed["Speed_sumsq"] = chunk["Speed"].pow(2).groupby([chunk[k] for k in group_keys]).sum()
            speed = speed.rename(columns={
                "count": "Speed_count", "sum": "Speed_sum",
                "min": "Speed_min", "max": "Speed_max",
            })
            parts.append(speed)

        if "Throttle" in chunk.columns:
            throttle = grouped["Throttle"].agg(["count", "sum"])
            throttle = throttle.rename(columns={"count": "Throttle_count", "sum": "Throttle_sum"})
            throttle_bins = chunk[group_keys + ["Throttle"]].dropna(subset=["Throttle"]).copy()
            if not throttle_bins.empty:
                throttle_bins["throttle_bin"] = throttle_bins["Throttle"].round().clip(0, 100).astype("int16")
                bin_counts = (
                    throttle_bins.groupby(group_keys + ["throttle_bin"], dropna=False)
                    .size()
                    .unstack("throttle_bin", fill_value=0)
                )
                bin_counts.columns = [f"throttle_bin_{int(c)}" for c in bin_counts.columns]
                throttle = throttle.join(bin_counts, how="left")
            parts.append(throttle)

        if "Brake" in chunk.columns:
            brake = grouped["Brake"].agg(["count", "sum"])
            brake = brake.rename(columns={"count": "Brake_count", "sum": "Brake_sum"})
            parts.append(brake)

        if "RPM" in chunk.columns:
            rpm = grouped["RPM"].agg(["count", "sum"])
            rpm["RPM_sumsq"] = chunk["RPM"].pow(2).groupby([chunk[k] for k in group_keys]).sum()
            rpm = rpm.rename(columns={"count": "RPM_count", "sum": "RPM_sum"})
            parts.append(rpm)

        if "nGear" in chunk.columns:
            gear = grouped["nGear"].agg(["count", "sum", "max"])
            gear = gear.rename(columns={
                "count": "nGear_count", "sum": "nGear_sum", "max": "nGear_max",
            })
            parts.append(gear)

        if "DRS" in chunk.columns:
            drs = grouped["DRS"].agg(["sum"])
            drs = drs.rename(columns={"sum": "DRS_sum"})
            parts.append(drs)

        if parts:
            partials.append(pd.concat(parts, axis=1))

    if not partials:
        return pd.DataFrame()

    partial_stats = pd.concat(partials)
    combine_spec = {
        col: ("min" if col == "Speed_min" else "max" if col in {"Speed_max", "nGear_max"} else "sum")
        for col in partial_stats.columns
    }
    stats = partial_stats.groupby(level=group_keys).agg(combine_spec)
    features = pd.DataFrame(index=stats.index)

    if "Speed_count" in stats.columns:
        features["speed_max"] = stats["Speed_max"]
        features["speed_min"] = stats["Speed_min"]
        speed_stats = _combine_count_sum_sumsq(stats.rename(columns={
            "Speed_count": "speed_count", "Speed_sum": "speed_sum", "Speed_sumsq": "speed_sumsq",
        }), "speed")
        features["speed_mean"] = speed_stats["speed_mean"]
        features["speed_std"] = speed_stats["speed_std"]

    if "Throttle_count" in stats.columns:
        throttle_stats = _combine_count_sum_sumsq(stats.rename(columns={
            "Throttle_count": "throttle_count", "Throttle_sum": "throttle_sum",
        }), "throttle")
        features["throttle_mean"] = throttle_stats["throttle_mean"]
        features["throttle_q90"] = _throttle_q90_from_counts(stats)

    if "Brake_count" in stats.columns:
        brake_stats = _combine_count_sum_sumsq(stats.rename(columns={
            "Brake_count": "brake_count", "Brake_sum": "brake_sum",
        }), "brake")
        features["brake_mean"] = brake_stats["brake_mean"]
        features["brake_sum"] = stats["Brake_sum"]

    if "RPM_count" in stats.columns:
        rpm_stats = _combine_count_sum_sumsq(stats.rename(columns={
            "RPM_count": "rpm_count", "RPM_sum": "rpm_sum", "RPM_sumsq": "rpm_sumsq",
        }), "rpm")
        features["rpm_mean"] = rpm_stats["rpm_mean"]
        features["rpm_std"] = rpm_stats["rpm_std"]

    if "nGear_count" in stats.columns:
        gear_stats = _combine_count_sum_sumsq(stats.rename(columns={
            "nGear_count": "ngear_count", "nGear_sum": "ngear_sum",
        }), "ngear")
        features["ngear_mean"] = gear_stats["ngear_mean"]
        features["ngear_max"] = stats["nGear_max"]

    if "DRS_sum" in stats.columns:
        features["drs_sum"] = stats["DRS_sum"]

    agg = features.reset_index()
    agg.columns = [c.lower() for c in agg.columns]
    agg = agg.sort_values(["session_key", "driver_number", "lap_number"])
    tele_feature_cols = [c for c in agg.columns if c not in group_keys]
    agg[tele_feature_cols] = agg.groupby(["session_key", "driver_number"])[tele_feature_cols].shift(1)
    agg = agg.rename(columns={col: f"tele_prev_{col}" for col in tele_feature_cols})

    logger.info(f"Telemetry aggregated: {len(agg):,} lap-level rows | {len(tele_feature_cols)} features (lag-1)")
    return agg


def aggregate_location_parquet_to_lap(path: Path, batch_size: int = 1_000_000) -> pd.DataFrame:
    """
    Aggregate cleaned/location.parquet to previous-lap spatial features.

    Uses only GPS-like coordinates, not the position endpoint. Features are shifted
    by one lap so lap N only sees trajectory information available through lap N-1.
    """
    if not path.exists():
        return pd.DataFrame()

    pf = pq.ParquetFile(path)
    schema_cols = set(pf.schema_arrow.names)
    group_keys = ["session_key", "driver_number", "lap_number"]
    coord_cols = [c for c in ["X", "Y", "Z"] if c in schema_cols]
    if not set(group_keys).issubset(schema_cols) or len(coord_cols) < 2:
        return pd.DataFrame()

    columns = group_keys + coord_cols
    partials = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        chunk = batch.to_pandas()
        if chunk.empty:
            continue
        for col in columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=group_keys)
        if chunk.empty:
            continue

        grouped = chunk.groupby(group_keys, dropna=False)
        parts = [grouped.size().rename("loc_points_count")]
        for col in coord_cols:
            coord = grouped[col].agg(["min", "max", "std"])
            coord.columns = [f"{col.lower()}_{stat}" for stat in coord.columns]
            parts.append(coord)
        partials.append(pd.concat(parts, axis=1))

    if not partials:
        return pd.DataFrame()

    partial_stats = pd.concat(partials)
    combine_spec = {}
    for col in partial_stats.columns:
        if col.endswith("_min"):
            combine_spec[col] = "min"
        elif col.endswith("_max"):
            combine_spec[col] = "max"
        elif col == "loc_points_count":
            combine_spec[col] = "sum"
        else:
            combine_spec[col] = "mean"

    stats = partial_stats.groupby(level=group_keys).agg(combine_spec).reset_index()
    if {"x_min", "x_max", "y_min", "y_max"}.issubset(stats.columns):
        stats["loc_bbox_area"] = (stats["x_max"] - stats["x_min"]).abs() * (stats["y_max"] - stats["y_min"]).abs()
    if {"x_std", "y_std"}.issubset(stats.columns):
        stats["loc_spread_xy"] = np.sqrt(stats["x_std"].pow(2) + stats["y_std"].pow(2))

    stats = stats.sort_values(group_keys)
    loc_feature_cols = [c for c in stats.columns if c not in group_keys]
    stats[loc_feature_cols] = stats.groupby(["session_key", "driver_number"])[loc_feature_cols].shift(1)
    stats = stats.rename(columns={col: f"loc_prev_{col}" for col in loc_feature_cols})

    logger.info(f"Location aggregated: {len(stats):,} lap-level rows | {len(loc_feature_cols)} features (lag-1)")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 8. Winsorization
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_engineering(config_path=None, dry_run: bool = False) -> pd.DataFrame:
    """
    Build lap-level master dataset.
    Granularity: (session_key, driver_number, lap_number)
    """
    config = load_config(config_path)
    setup_logging()
    c = config.cleaned_dir

    laps = _read(c / "laps.csv")
    if laps.empty:
        raise RuntimeError("laps.csv missing — run clean_data first")

    sessions  = _read(c / "sessions.csv")
    drivers   = _read(c / "drivers.csv")
    stints    = _read(c / "stints.csv")
    weather   = _read(c / "weather.csv")
    results   = _read(c / "session_results.csv")
    pit       = _read(c / "pit.csv")
    race_control = _read(c / "race_control.csv")
    overtakes = _read(c / "overtakes.csv")
    car_data_path = c / "car_data.parquet"  # Telemetry — có thể rỗng nếu chưa crawl
    location_path = c / "location.parquet"

    logger.info(f"Base laps: {len(laps):,} rows")

    # ── Join session metadata ──────────────────────────────────────────────
    if not sessions.empty:
        s_cols = [c for c in ["session_key", "year", "session_type", "circuit_key",
                               "circuit_short_name", "country_name", "date_start", "date_end"]
                  if c in sessions.columns]
        laps = laps.merge(sessions[s_cols].drop_duplicates("session_key"),
                          on="session_key", how="left", suffixes=("", "_sess"))

    # ── Join driver metadata ───────────────────────────────────────────────
    if not drivers.empty:
        d_cols = [c for c in ["session_key", "driver_number", "full_name", "name_acronym", "team_name"]
                  if c in drivers.columns]
        laps = laps.merge(drivers[d_cols].drop_duplicates(["session_key", "driver_number"]),
                          on=["session_key", "driver_number"], how="left")

    # ── Learn tyre cliffs ──────────────────────────────────────────────────
    cliffs = learn_tyre_cliffs(stints, laps)
    logger.info(f"Tyre cliffs: {cliffs}")

    # ── Feature steps ──────────────────────────────────────────────────────
    laps = add_stint_features(laps, stints, cliffs)
    logger.info("Stint features added")

    laps = add_rolling_features(laps)
    logger.info("Rolling features added")

    laps = add_weather_features(laps, weather)
    logger.info("Weather features added")

    laps = add_pit_features(laps, pit)
    logger.info("Pit strategy features added")

    laps = add_race_control_features(laps, race_control)
    logger.info("Race-control features added")

    laps = add_overtake_features(laps, overtakes)
    logger.info("Overtake features added")

    laps = add_targets(laps, results)
    logger.info("Targets added")

    # ── Telemetry aggregation (Gold layer) — đọc parquet theo batch để tránh hết RAM ──────
    if car_data_path.exists():
        tele_agg = aggregate_telemetry_parquet_to_lap(car_data_path)
        if not tele_agg.empty:
            laps = laps.merge(tele_agg, on=["session_key", "driver_number", "lap_number"], how="left")
            tele_coverage = laps["tele_prev_speed_max"].notna().mean() if "tele_prev_speed_max" in laps.columns else 0
            logger.info(f"Telemetry joined | coverage: {tele_coverage:.1%} of laps")
            if tele_coverage < 0.3:
                logger.warning("Telemetry coverage < 30% — dữ liệu telemetry có thể quá ít. Kiểm tra data/raw/car_data/")
        else:
            logger.info("Telemetry (car_data) không có cột cần thiết hoặc không aggregate được — bỏ qua.")
    else:
        logger.info("Telemetry (car_data) không có — bỏ qua aggregate. Chạy crawler để có dữ liệu.")

    if location_path.exists():
        loc_agg = aggregate_location_parquet_to_lap(location_path)
        if not loc_agg.empty:
            laps = laps.merge(loc_agg, on=["session_key", "driver_number", "lap_number"], how="left")
            loc_coverage = laps["loc_prev_loc_points_count"].notna().mean() if "loc_prev_loc_points_count" in laps.columns else 0
            logger.info(f"Location joined | coverage: {loc_coverage:.1%} of laps")
        else:
            logger.info("Location data has no usable lap-level coordinate columns; skipping.")
    else:
        logger.info("Location parquet not found; skipping location features.")

    laps = winsorize(laps)
    logger.info("Winsorization done")

    # ── Final cleanup ──────────────────────────────────────────────────────
    laps = laps.drop_duplicates(subset=["session_key", "driver_number", "lap_number"])
    laps = laps.reset_index(drop=True)
    laps = move_outcomes_to_end(laps)

    logger.info(f"Master dataset: {len(laps):,} rows × {len(laps.columns)} cols")
    logger.info(f"Win rate: {laps['target_win'].mean():.3f} ({laps['target_win'].sum()} wins / {len(laps)} laps)")

    if dry_run:
        logger.info("[Dry Run] Skipping save")
        return laps

    write_csv(laps, config.processed_dir / "master_dataset.csv")
    try:
        laps.to_parquet(config.processed_dir / "master_dataset.parquet", index=False)
    except Exception as exc:
        logger.warning(f"Parquet save failed: {exc}")

    write_json({
        "built_at": utc_now_iso(),
        "rows": len(laps),
        "cols": len(laps.columns),
        "tyre_cliffs_used": cliffs,
        "prediction_contract": "live_before_lap_n",
        "grain": ["session_key", "driver_number", "lap_number"],
        "model_feature_columns": model_feature_columns(laps),
        "win_rate": float(laps["target_win"].mean()) if "target_win" in laps.columns else None,
    }, config.metadata_dir / "feature_engineering_metadata.json")

    return laps


if __name__ == "__main__":
    setup_logging()
    build_feature_engineering()

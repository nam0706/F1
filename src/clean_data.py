import os
import sys
import logging
import shutil
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

BASE_DIR       = os.path.join(os.path.dirname(__file__), "..")
RAW_DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")


# Táº¥t cáº£ cÃ¡c session type há»£p lá»‡ (Ä‘á»“ng bá»™ vá»›i pipeline_config.yaml)
VALID_SESSION_TYPES = {"Race", "Sprint", "Qualifying", "Sprint Shootout", "Sprint Qualifying"}


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    if "session_type" in df.columns:
        df = df[df["session_type"].isin(VALID_SESSION_TYPES)].copy()

    for col in ["date_start", "date_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ["year", "session_key", "circuit_key"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "circuit_short_name" in df.columns:
        if df["circuit_short_name"].isna().all():
            df.drop(columns=["circuit_short_name"], inplace=True)
        else:
            df["circuit_short_name"] = df["circuit_short_name"].fillna("Unknown")

    for col in ["location", "country_name", "gmt_offset"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    df.drop_duplicates(subset=["session_key"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_drivers(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    for col in ["driver_number", "session_key"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["full_name", "name_acronym", "team_name", "country_code"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": "Unknown", "": "Unknown"})

    if "headshot_url" in df.columns:
        default = "https://media.formula1.com/d_driver_fallback_image.png"
        df["headshot_url"] = df["headshot_url"].fillna(default).replace({"": default})

    df.drop_duplicates(subset=["driver_number", "session_key"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_session_results(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    for col in ["session_key", "driver_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        df = df[df["position"].between(1, 25) | df["position"].isna()].copy()

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().replace({"nan": "Unknown", "": "Unknown"})

    df.dropna(subset=["session_key", "driver_number"], inplace=True)
    df.drop_duplicates(subset=["session_key", "driver_number"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    for col in ["session_key", "driver_number", "lap_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================================================
    # BÆ¯á»šC 1: XÃC Äá»ŠNH & LOáº I Bá»Ž DNF (Did Not Finish)
    # NguyÃªn táº¯c: KHÃ”NG BAO GIá»œ Ä‘iá»n lap_duration báº±ng median.
    # Náº¿u lap_duration lÃ  NaN â†’ xe khÃ´ng hoÃ n thÃ nh vÃ²ng Ä‘ua â†’
    # Ä‘iá»n sá»‘ giáº£ vÃ o sáº½ khiáº¿n AI tÆ°á»Ÿng xe váº«n Ä‘ang cháº¡y bÃ¬nh thÆ°á»ng.
    # =========================================================
    if "lap_duration" in df.columns:
        df["lap_duration"] = pd.to_numeric(df["lap_duration"], errors="coerce")

        # 1a. Loáº¡i bá» hoÃ n toÃ n cÃ¡c vÃ²ng khÃ´ng hoÃ n thÃ nh (DNF/SC/lap bá»‹ há»§y)
        dnf_count = df["lap_duration"].isna().sum()
        df = df.dropna(subset=["lap_duration"]).copy()
        if dnf_count:
            logger.info(f"  Removed {dnf_count} incomplete laps (lap_duration=NaN â†’ likely DNF/SC)")

        # 1b. Lá»c hard bound: giÃ¡ trá»‹ ngoÃ i [30s, 600s] lÃ  lá»—i sensor, khÃ´ng pháº£i DNF
        out_of_range = (~df["lap_duration"].between(30, 600)).sum()
        df = df[df["lap_duration"].between(30, 600)].copy()
        if out_of_range:
            logger.info(f"  Removed {out_of_range} laps with sensor-error duration (outside 30-600s)")

        # 1c. Z-score flagging trÃªn CÃC VÃ’NG ÄÃƒ HOÃ€N THÃ€NH
        # VÃ²ng |Z| > 3 khÃ´ng bá»‹ xÃ³a â€” gáº¯n nhÃ£n Ä‘á»ƒ AI phÃ¢n biá»‡t "vÃ²ng sá»± cá»‘ chiáº¿n thuáº­t"
        if {"session_key", "driver_number"}.issubset(df.columns):
            grp = df.groupby(["session_key", "driver_number"])["lap_duration"]
            mean = grp.transform("mean")
            std  = grp.transform("std").replace(0, np.nan)
            df["lap_zscore"] = ((df["lap_duration"] - mean) / std).abs()
            df["is_outlier_lap"] = df["lap_zscore"] > 3
            df.drop(columns=["lap_zscore"], inplace=True)
            logger.info(f"  Flagged {df['is_outlier_lap'].sum()} statistical outlier laps (|Z|>3, kept for AI context)")

    # =========================================================
    # BÆ¯á»šC 2: CHá»ˆ SAU KHI ÄÃƒ XÃC NHáº¬N VÃ’NG HOÃ€N THÃ€NH
    # má»›i impute cÃ¡c cá»™t cáº£m biáº¿n phá»¥ (tá»‘c Ä‘á»™, sector time...)
    # LÃ½ do: Ä‘Ã¢y lÃ  "lá»—i cáº£m biáº¿n" trÃªn vÃ²ng Ä‘ua thá»±c sá»± tá»“n táº¡i,
    # khÃ´ng pháº£i dá»¯ liá»‡u cá»§a vÃ²ng khÃ´ng hoÃ n thÃ nh.
    # =========================================================
    for col in ["i1_speed", "i2_speed", "st_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~df[col].between(0, 400), col] = np.nan
            if {"session_key", "driver_number"}.issubset(df.columns):
                driver_median = df.groupby(["session_key", "driver_number"])[col].transform("median")
                global_median = df[col].median()
                df[col] = df[col].fillna(driver_median).fillna(global_median)
            else:
                df[col] = df[col].fillna(df[col].median())

    for col in ["duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if {"session_key", "driver_number"}.issubset(df.columns):
                driver_median = df.groupby(["session_key", "driver_number"])[col].transform("median")
                global_median = df[col].median()
                df[col] = df[col].fillna(driver_median).fillna(global_median)
            else:
                df[col] = df[col].fillna(df[col].median())

    for col in ["segments_sector_1", "segments_sector_2", "segments_sector_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
        # CHá»NG Lá»–I MERGE_ASOF: Loáº¡i bá» laps khÃ´ng cÃ³ má»‘c thá»i gian ngay tá»« táº§ng Silver
        initial_laps = len(df)
        df = df.dropna(subset=["date_start"]).copy()
        dropped = initial_laps - len(df)
        if dropped > 0:
            logger.warning(f"  clean_laps: Loáº¡i bá» {dropped} laps bá»‹ lá»—i date_start (NaT)")

    if "is_pit_out_lap" in df.columns:
        df["is_pit_out_lap"] = df["is_pit_out_lap"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )

    df.dropna(subset=["session_key", "driver_number", "lap_number"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    if "session_key" in df.columns:
        df["session_key"] = pd.to_numeric(df["session_key"], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df.dropna(subset=["date"], inplace=True)

    numeric_bounds = {
        "air_temperature":   (-30, 60),
        "track_temperature": (-10, 80),
        "humidity":          (0, 100),
        "pressure":          (800, 1100),
        "wind_speed":        (0, 200),
        "wind_direction":    (0, 360),
        "rainfall":          (0, 1),
    }
    for col, (lo, hi) in numeric_bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~df[col].between(lo, hi), col] = np.nan
            df[col] = df[col].fillna(df[col].median())

    df.dropna(subset=["session_key"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_stints(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    for col in ["session_key", "driver_number", "stint_number", "lap_start", "lap_end", "tyre_age_at_start"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "lap_start" in df.columns and "lap_end" in df.columns:
        df = df[df["lap_end"] >= df["lap_start"]].copy()

    if "compound" in df.columns:
        valid = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"}
        df["compound"] = df["compound"].astype(str).str.strip().str.upper()
        df["compound"] = df["compound"].apply(lambda x: x if x in valid else "UNKNOWN")

    df.dropna(subset=["session_key", "driver_number", "stint_number"], inplace=True)
    df.drop_duplicates(subset=["session_key", "driver_number", "stint_number"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_starting_grid(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    for col in ["session_key", "driver_number", "position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "position" in df.columns:
        df = df[df["position"].between(1, 25) | df["position"].isna()].copy()

    df.dropna(subset=["session_key", "driver_number"], inplace=True)
    df.drop_duplicates(subset=["session_key", "driver_number"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(how="all", inplace=True)

    for col in ["session_key", "driver_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df.dropna(subset=["date"], inplace=True)

    for col in ["gap_to_leader", "interval"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
            lap_mask = df[col].str.contains("LAP", case=False, na=False)
            df[col + "_lapped"] = lap_mask
            df.loc[lap_mask, col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] > 300, col] = np.nan

    df.dropna(subset=["session_key", "driver_number"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_position(df: pd.DataFrame) -> pd.DataFrame:
    """Clean position time-series data (nhiá»u dÃ²ng/driver/session theo thá»i gian thá»±c)."""
    df.dropna(how="all", inplace=True)

    for col in ["session_key", "driver_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df.dropna(subset=["date"], inplace=True)

    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        df = df[df["position"].between(1, 25) | df["position"].isna()].copy()

    df.dropna(subset=["session_key", "driver_number"], inplace=True)
    # KhÃ´ng drop_duplicates vÃ¬ Ä‘Ã¢y lÃ  time-series (nhiá»u dÃ²ng má»—i driver)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean car telemetry data (Silver layer).
    NHIá»†M Vá»¤: Lá»c rÃ¡c ká»¹ thuáº­t, Ã©p kiá»ƒu dá»¯ liá»‡u.
    GIá»® NGUYÃŠN: dá»¯ liá»‡u váº«n á»Ÿ cáº¥p Ä‘á»™ má»-giÃ¢y (khÃ´ng aggregate).
    """
    df.dropna(how="all", inplace=True)

    # Ã‰p kiá»ƒu cÃ¡c cá»™t quan trá»ng
    numeric_cols = ["Speed", "RPM", "Throttle", "Brake", "DRS", "nGear",
                    "X", "Y", "Z"]  # FastF1 column names
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Lá»c cÃ¡c giÃ¡ trá»‹ váº­t lÃ½ phi lÃ½ (lá»—i cáº£m biáº¿n)
    if "Speed" in df.columns:
        out_of_range = (~df["Speed"].between(0, 400)).sum()
        df = df[df["Speed"].between(0, 400) | df["Speed"].isna()].copy()
        if out_of_range:
            logger.info(f"  car_data: removed {out_of_range} rows with Speed outside [0, 400] km/h")

    if "RPM" in df.columns:
        df.loc[~df["RPM"].between(0, 20000), "RPM"] = np.nan

    if "Throttle" in df.columns:
        df.loc[~df["Throttle"].between(0, 100), "Throttle"] = np.nan

    # Time column
    if "Time" in df.columns:
        df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)

    # Báº¯t buá»™c cÃ³ driver_number
    if "driver_number" in df.columns:
        df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
        df.dropna(subset=["driver_number"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean GPS location data (Silver layer).
    NHIá»†M Vá»¤: Lá»c tá»a Ä‘á»™ phi lÃ½, Ã©p kiá»ƒu dá»¯ liá»‡u.
    GIá»® NGUYÃŠN: dá»¯ liá»‡u váº«n á»Ÿ cáº¥p Ä‘á»™ má»-giÃ¢y.
    """
    df.dropna(how="all", inplace=True)

    for col in ["X", "Y", "Z"]:  # GPS coordinates
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df.dropna(subset=["Date"], inplace=True)

    if "driver_number" in df.columns:
        df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
        df.dropna(subset=["driver_number"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Cleaner v2: EDA-aligned Silver layer cleaners.
# The legacy cleaners above are intentionally kept for reference/backward use.
# CLEAN_FUNCTIONS below dispatches to this v2 set.
# ---------------------------------------------------------------------------

def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all").copy()


def _to_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_datetime_utc(df: pd.DataFrame, cols) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _clean_text(df: pd.DataFrame, cols, default="Unknown") -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna(default)
                .astype(str)
                .str.strip()
                .replace({"nan": default, "NaN": default, "": default})
            )
    return df


def _keep_best_duplicate(df: pd.DataFrame, keys) -> pd.DataFrame:
    keys = [col for col in keys if col in df.columns]
    if not keys:
        return df
    completeness = df.notna().sum(axis=1)
    return (
        df.assign(_completeness=completeness)
        .sort_values("_completeness", ascending=False)
        .drop_duplicates(subset=keys, keep="first")
        .drop(columns="_completeness")
    )


def clean_sessions_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_datetime_utc(df, ["date_start", "date_end"])
    df = _to_numeric(df, ["year", "meeting_key", "session_key", "circuit_key"])
    _clean_text(df, ["session_name", "session_type", "location", "country_name", "gmt_offset"])

    if "session_type" in df.columns:
        df = df[df["session_type"].isin(VALID_SESSION_TYPES)].copy()

    df = _keep_best_duplicate(df, ["session_key"])
    df.dropna(subset=[col for col in ["session_key"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_meetings_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_datetime_utc(df, ["date_start", "date_end"])
    df = _to_numeric(df, ["meeting_key", "country_key", "circuit_key", "year"])
    _clean_text(
        df,
        [
            "meeting_name",
            "meeting_official_name",
            "location",
            "country_code",
            "country_name",
            "circuit_short_name",
            "circuit_type",
            "gmt_offset",
        ],
    )

    if "is_cancelled" in df.columns:
        df["is_cancelled"] = df["is_cancelled"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )

    df = _keep_best_duplicate(df, ["meeting_key", "year"])
    df.dropna(subset=[col for col in ["meeting_key"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_drivers_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["driver_number", "session_key", "meeting_key"])
    _clean_text(df, ["full_name", "name_acronym", "team_name", "country_code"])

    if "headshot_url" in df.columns:
        default = "https://media.formula1.com/d_driver_fallback_image.png"
        df["headshot_url"] = df["headshot_url"].fillna(default).replace({"": default})

    df.dropna(subset=[col for col in ["session_key", "driver_number"] if col in df.columns], inplace=True)
    df = _keep_best_duplicate(df, ["session_key", "driver_number"])
    return df.reset_index(drop=True)


def clean_session_results_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(
        df,
        [
            "position",
            "driver_number",
            "number_of_laps",
            "meeting_key",
            "session_key",
            "points",
        ],
    )
    _clean_text(df, ["status"], default="Unknown")

    if "position" in df.columns:
        invalid = ~df["position"].between(1, 25) & df["position"].notna()
        df["is_invalid_position"] = invalid
        df.loc[invalid, "position"] = np.nan

    for col in ["dnf", "dns", "dsq"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )

    df.dropna(subset=[col for col in ["session_key", "driver_number"] if col in df.columns], inplace=True)
    df = _keep_best_duplicate(df, ["session_key", "driver_number"])
    return df.reset_index(drop=True)


def clean_laps_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(
        df,
        [
            "session_key",
            "meeting_key",
            "driver_number",
            "lap_number",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
            "i1_speed",
            "i2_speed",
            "st_speed",
        ],
    )
    df = _to_datetime_utc(df, ["date_start"])

    if "lap_duration" in df.columns:
        df["lap_duration"] = pd.to_numeric(df["lap_duration"], errors="coerce")
        df["is_missing_lap_duration"] = df["lap_duration"].isna()
        df["is_invalid_lap_duration"] = (
            df["lap_duration"].notna() & ~df["lap_duration"].between(30, 600)
        )
        df.loc[df["is_invalid_lap_duration"], "lap_duration"] = np.nan

        if {"session_key", "driver_number"}.issubset(df.columns):
            grp = df.groupby(["session_key", "driver_number"])["lap_duration"]
            mean = grp.transform("mean")
            std = grp.transform("std").replace(0, np.nan)
            zscore = ((df["lap_duration"] - mean) / std).abs()
            df["is_outlier_lap"] = (zscore > 3).fillna(False)
        else:
            df["is_outlier_lap"] = False

    for col in ["i1_speed", "i2_speed", "st_speed"]:
        if col in df.columns:
            df[f"is_invalid_{col}"] = df[col].notna() & ~df[col].between(0, 400)
            df.loc[df[f"is_invalid_{col}"], col] = np.nan
            if {"session_key", "driver_number"}.issubset(df.columns):
                driver_median = df.groupby(["session_key", "driver_number"])[col].transform("median")
                df[col] = df[col].fillna(driver_median).fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].median())

    for col in ["duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        if col in df.columns:
            df[f"is_invalid_{col}"] = df[col].notna() & ~df[col].between(5, 120)
            df.loc[df[f"is_invalid_{col}"], col] = np.nan
            if {"session_key", "driver_number"}.issubset(df.columns):
                driver_median = df.groupby(["session_key", "driver_number"])[col].transform("median")
                df[col] = df[col].fillna(driver_median).fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].median())

    for col in ["segments_sector_1", "segments_sector_2", "segments_sector_3"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "is_pit_out_lap" in df.columns:
        df["is_pit_out_lap"] = df["is_pit_out_lap"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )

    df.dropna(subset=[col for col in ["session_key", "driver_number", "lap_number", "date_start"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_weather_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["session_key", "meeting_key"])
    df = _to_datetime_utc(df, ["date"])

    numeric_bounds = {
        "air_temperature": (-30, 60),
        "track_temperature": (-10, 80),
        "humidity": (0, 100),
        "pressure": (800, 1100),
        "wind_speed": (0, 200),
        "wind_direction": (0, 360),
        "rainfall": (0, 1),
    }
    for col, (lo, hi) in numeric_bounds.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        flag_col = f"is_invalid_{col}"
        df[flag_col] = df[col].notna() & ~df[col].between(lo, hi)
        df.loc[df[flag_col], col] = np.nan

        if "session_key" in df.columns:
            session_median = df.groupby("session_key")[col].transform("median")
            df[col] = df[col].fillna(session_median)
        if "meeting_key" in df.columns:
            meeting_median = df.groupby("meeting_key")[col].transform("median")
            df[col] = df[col].fillna(meeting_median)
        df[col] = df[col].fillna(df[col].median())

    df.dropna(subset=[col for col in ["session_key", "date"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_stints_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(
        df,
        ["session_key", "meeting_key", "driver_number", "stint_number", "lap_start", "lap_end", "tyre_age_at_start"],
    )

    if "lap_start" in df.columns and "lap_end" in df.columns:
        df["is_invalid_stint_lap_range"] = df["lap_end"].notna() & df["lap_start"].notna() & (df["lap_end"] < df["lap_start"])
        df = df[~df["is_invalid_stint_lap_range"]].copy()

    if "compound" in df.columns:
        valid = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"}
        df["compound"] = df["compound"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        df["compound"] = df["compound"].where(df["compound"].isin(valid), "UNKNOWN")

    df.dropna(subset=[col for col in ["session_key", "driver_number", "stint_number"] if col in df.columns], inplace=True)
    df = _keep_best_duplicate(df, ["session_key", "driver_number", "stint_number"])
    return df.reset_index(drop=True)


def clean_starting_grid_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["session_key", "meeting_key", "driver_number", "position"])
    if "position" in df.columns:
        df["is_invalid_grid_position"] = df["position"].notna() & ~df["position"].between(1, 25)
        df.loc[df["is_invalid_grid_position"], "position"] = np.nan
    df.dropna(subset=[col for col in ["session_key", "driver_number"] if col in df.columns], inplace=True)
    df = _keep_best_duplicate(df, ["session_key", "driver_number"])
    return df.reset_index(drop=True)


def clean_intervals_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["session_key", "meeting_key", "driver_number"])
    df = _to_datetime_utc(df, ["date"])

    for col in ["gap_to_leader", "interval"]:
        if col in df.columns:
            raw = df[col].astype(str).str.strip().replace({"nan": np.nan, "NaN": np.nan, "": np.nan})
            lapped = raw.str.contains("LAP", case=False, na=False)
            df[f"{col}_lapped"] = lapped
            raw = raw.mask(lapped, np.nan)
            df[col] = pd.to_numeric(raw, errors="coerce")
            df[f"is_invalid_{col}"] = df[col].notna() & ~df[col].between(-1, 300)
            df.loc[df[f"is_invalid_{col}"], col] = np.nan

    df.dropna(subset=[col for col in ["session_key", "driver_number", "date"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_position_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["session_key", "meeting_key", "driver_number", "position"])
    df = _to_datetime_utc(df, ["date"])
    if "position" in df.columns:
        df["is_invalid_position"] = df["position"].notna() & ~df["position"].between(1, 25)
        df.loc[df["is_invalid_position"], "position"] = np.nan
    df.dropna(subset=[col for col in ["session_key", "driver_number", "date"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_race_control_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["meeting_key", "session_key", "driver_number", "lap_number", "sector", "qualifying_phase"])
    df = _to_datetime_utc(df, ["date"])
    _clean_text(df, ["category", "flag", "scope", "message"], default="")

    if "lap_number" in df.columns:
        invalid = df["lap_number"].notna() & ~df["lap_number"].between(1, 200)
        df["is_invalid_lap_number"] = invalid
        df["lap_number_clean"] = df["lap_number"].where(~invalid, np.nan)

    df.dropna(subset=[col for col in ["session_key", "date"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_pit_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(
        df,
        ["meeting_key", "session_key", "driver_number", "lap_number", "lane_duration", "stop_duration", "pit_duration"],
    )
    df = _to_datetime_utc(df, ["date"])

    for col in ["lane_duration", "stop_duration", "pit_duration"]:
        if col in df.columns:
            df[f"is_invalid_{col}"] = df[col].notna() & ~df[col].between(0, 300)
            df.loc[df[f"is_invalid_{col}"], col] = np.nan

    df.dropna(subset=[col for col in ["session_key", "driver_number"] if col in df.columns], inplace=True)
    df = _keep_best_duplicate(df, ["session_key", "driver_number", "lap_number", "date"])
    return df.reset_index(drop=True)


def clean_overtakes_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["meeting_key", "session_key", "driver_number", "overtaken_driver_number", "lap_number"])
    df = _to_datetime_utc(df, ["date"])
    if "lap_number" in df.columns:
        df["is_invalid_lap_number"] = df["lap_number"].notna() & ~df["lap_number"].between(1, 200)
        df.loc[df["is_invalid_lap_number"], "lap_number"] = np.nan
    df.dropna(subset=[col for col in ["session_key"] if col in df.columns], inplace=True)
    df = _keep_best_duplicate(df, [col for col in ["session_key", "driver_number", "overtaken_driver_number", "lap_number", "date"] if col in df.columns])
    return df.reset_index(drop=True)


def clean_team_radio_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["meeting_key", "session_key", "driver_number"])
    df = _to_datetime_utc(df, ["date"])
    _clean_text(df, ["recording_url"], default="")
    df.dropna(subset=[col for col in ["session_key", "driver_number", "date"] if col in df.columns], inplace=True)
    df = _keep_best_duplicate(df, ["session_key", "driver_number", "date", "recording_url"])
    return df.reset_index(drop=True)


def clean_car_data_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(
        df,
        ["session_key", "meeting_key", "driver_number", "lap_number", "Speed", "RPM", "Throttle", "Brake", "DRS", "nGear", "X", "Y", "Z"],
    )
    df = _to_datetime_utc(df, ["Date"])
    if "Time" in df.columns:
        df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")

    bounds = {
        "Speed": (0, 400),
        "RPM": (0, 20000),
        "Throttle": (0, 100),
        "Brake": (0, 1),
        "DRS": (0, 20),
        "nGear": (0, 8),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df[f"is_invalid_{col}"] = df[col].notna() & ~df[col].between(lo, hi)
            df.loc[df[f"is_invalid_{col}"], col] = np.nan

    df.dropna(subset=[col for col in ["driver_number"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


def clean_location_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_empty_rows(df)
    df = _to_numeric(df, ["session_key", "meeting_key", "driver_number", "lap_number", "X", "Y", "Z"])
    df = _to_datetime_utc(df, ["Date"])
    if "Time" in df.columns:
        df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")
    df.dropna(subset=[col for col in ["driver_number"] if col in df.columns], inplace=True)
    return df.reset_index(drop=True)


CLEAN_FUNCTIONS = {
    "sessions.csv": clean_sessions_v2,
    "meetings.csv": clean_meetings_v2,
    "drivers.csv": clean_drivers_v2,
    # Raw endpoint is session_result; downstream expects cleaned session_results.csv.
    "session_results.csv": {"source_endpoint": "session_result", "clean_fn": clean_session_results_v2},
    "laps.csv": clean_laps_v2,
    "weather.csv": clean_weather_v2,
    "stints.csv": clean_stints_v2,
    "starting_grid.csv": clean_starting_grid_v2,
    "intervals.csv": clean_intervals_v2,
    "position.csv": clean_position_v2,
    "race_control.csv": clean_race_control_v2,
    "pit.csv": clean_pit_v2,
    "overtakes.csv": clean_overtakes_v2,
    "team_radio.csv": clean_team_radio_v2,
    # Telemetry Silver layer: filter physical garbage, keep sub-second grain.
    "car_data.parquet": clean_car_data_v2,
    "location.parquet": clean_location_v2,
}


LARGE_ENDPOINTS = {"car_data", "location", "laps", "intervals", "position"}


def _read_raw_file(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def _clean_single_file(path, clean_fn):
    df = _read_raw_file(path)
    raw_rows = len(df)
    if df.empty:
        return pd.DataFrame(), raw_rows
    return clean_fn(df), raw_rows


def _part_path(parts_dir, source_path, index, is_parquet_output):
    suffix = ".parquet" if is_parquet_output else ".csv"
    safe_stem = f"{index:05d}_{source_path.stem}".replace(" ", "_")
    return parts_dir / f"{safe_stem}{suffix}"


def _write_clean_part(df, part_path, is_parquet_output):
    part_path.parent.mkdir(parents=True, exist_ok=True)
    if is_parquet_output:
        df.to_parquet(part_path, index=False, compression="zstd")
    else:
        df.to_csv(part_path, index=False)


def _merge_parquet_parts(part_paths, out_path):
    import pyarrow.parquet as pq

    writer = None
    try:
        for part_path in part_paths:
            table = pq.read_table(part_path)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _merge_csv_parts(part_paths, out_path):
    columns = []
    seen = set()
    for part_path in part_paths:
        header = pd.read_csv(part_path, nrows=0).columns.tolist()
        for col in header:
            if col not in seen:
                seen.add(col)
                columns.append(col)

    first = True
    for part_path in part_paths:
        for chunk in pd.read_csv(part_path, chunksize=200_000, low_memory=False):
            chunk = chunk.reindex(columns=columns)
            chunk.to_csv(out_path, index=False, mode="a", header=first)
            first = False


def _merge_clean_parts(part_paths, out_path, is_parquet_output):
    if out_path.exists():
        out_path.unlink()
    if is_parquet_output:
        _merge_parquet_parts(part_paths, out_path)
    else:
        _merge_csv_parts(part_paths, out_path)


def _clean_files_to_parts(all_files, clean_fn, parts_dir, is_parquet_output):
    raw_rows = 0
    clean_rows = 0
    part_paths = []

    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(all_files):
        try:
            df_cleaned, file_raw_rows = _clean_single_file(path, clean_fn)
            raw_rows += file_raw_rows
            if df_cleaned.empty:
                continue

            clean_rows += len(df_cleaned)
            part_path = _part_path(parts_dir, path, index, is_parquet_output)
            _write_clean_part(df_cleaned, part_path, is_parquet_output)
            part_paths.append(part_path)
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")

    return part_paths, raw_rows, clean_rows


def _clean_files_count_only(all_files, clean_fn):
    raw_rows = 0
    clean_rows = 0
    kept_files = 0

    for path in all_files:
        try:
            df_cleaned, file_raw_rows = _clean_single_file(path, clean_fn)
            raw_rows += file_raw_rows
            if df_cleaned.empty:
                continue
            clean_rows += len(df_cleaned)
            kept_files += 1
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")

    return kept_files, raw_rows, clean_rows


def _clean_small_endpoint(all_files, clean_fn):
    frames = []
    for path in all_files:
        try:
            df = _read_raw_file(path)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")

    if not frames:
        return pd.DataFrame(), 0, 0

    df_combined = pd.concat(frames, ignore_index=True)
    raw_rows = len(df_combined)
    df_cleaned = clean_fn(df_combined)
    return df_cleaned, raw_rows, len(df_cleaned)


def clean_all(dry_run: bool = False):
    print("=" * 60)
    print("   F1 DATA CLEANING  (clean_data.py)")
    if dry_run:
        print("   !!! DRY RUN MODE - No files will be saved !!!")
    print("   Raw -> data/raw/  |  Clean -> data/cleaned/")
    print("=" * 60)

    from pathlib import Path
    import glob

    raw_dir_path = Path(RAW_DATA_DIR)
    clean_dir_path = Path(CLEAN_DATA_DIR)
    clean_dir_path.mkdir(parents=True, exist_ok=True)
    
    summary = []

    for filename, cleaner_spec in CLEAN_FUNCTIONS.items():
        # Há»— trá»£ cáº£ .csv vÃ  .parquet â€” láº¥y tÃªn endpoint chÃ­nh xÃ¡c
        if isinstance(cleaner_spec, dict):
            clean_fn = cleaner_spec["clean_fn"]
            endpoint = cleaner_spec.get("source_endpoint")
        else:
            clean_fn = cleaner_spec
            endpoint = None

        is_parquet_output = filename.endswith(".parquet")
        if endpoint is None:
            endpoint = filename.replace(".parquet", "").replace(".csv", "")
        endpoint_dir = raw_dir_path / endpoint
        
        # Find all session-based CSVs and Parquets
        all_files = list(endpoint_dir.glob("**/*.csv")) + list(endpoint_dir.glob("**/*.parquet"))
        
        # Check for old flat file
        old_flat_path = raw_dir_path / filename
        if old_flat_path.exists():
            all_files.append(old_flat_path)

        if not all_files:
            logger.info(f"Skipping {filename} - No raw data found")
            summary.append({"file": filename, "status": "skipped"})
            continue

        logger.info(f"Cleaning {endpoint} (combining {len(all_files)} files)")

        # --- Cáº£i tiáº¿n B: Xá»­ lÃ½ tá»«ng file riÃªng láº» Ä‘á»ƒ tiáº¿t kiá»‡m RAM ---
        # Thay vÃ¬ náº¡p táº¥t cáº£ vÃ o RAM rá»“i concat, ta clean tá»«ng file trÆ°á»›c rá»“i má»›i ghÃ©p
        is_large_endpoint = endpoint in LARGE_ENDPOINTS
        out_path = clean_dir_path / filename

        if is_large_endpoint and dry_run:
            kept_files, raw_rows, clean_rows = _clean_files_count_only(all_files, clean_fn)
            if not kept_files:
                logger.info(f"Skipping {filename} - no rows remained after cleaning")
                summary.append({"file": filename, "status": "empty_after_clean"})
                continue
        elif is_large_endpoint:
            parts_dir = clean_dir_path / "_parts" / endpoint
            part_paths, raw_rows, clean_rows = _clean_files_to_parts(
                all_files,
                clean_fn,
                parts_dir,
                is_parquet_output,
            )
            if not part_paths:
                logger.info(f"Skipping {filename} - no rows remained after cleaning")
                summary.append({"file": filename, "status": "empty_after_clean"})
                continue
        else:
            df_cleaned, raw_rows, clean_rows = _clean_small_endpoint(all_files, clean_fn)
            if df_cleaned.empty:
                logger.info(f"Skipping {filename} - no rows remained after cleaning")
                summary.append({"file": filename, "status": "empty_after_clean"})
                continue
        
        if dry_run:
            logger.info(f"  [Dry Run] Would save {clean_rows} cleaned rows to {filename}")
        elif is_large_endpoint:
            _merge_clean_parts(part_paths, out_path, is_parquet_output)
        elif is_parquet_output:
            df_cleaned.to_parquet(out_path, index=False, compression="zstd")
        else:
            df_cleaned.to_csv(out_path, index=False)
        dropped = raw_rows - clean_rows
        logger.info(f"  {endpoint}: {raw_rows:,} -> {clean_rows:,} (dropped {dropped:,})")
        summary.append({
            "file": filename, 
            "raw_rows": raw_rows, 
            "clean_rows": clean_rows, 
            "dropped": dropped, 
            "status": "ok"
        })

    if not dry_run:
        summary_df = pd.DataFrame(summary)
        reports_dir = os.path.join(BASE_DIR, "reports", "data_quality")
        os.makedirs(reports_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(reports_dir, "cleaning_summary.csv"), index=False)
        summary_df.to_markdown(os.path.join(reports_dir, "cleaning_summary.md"), index=False)

        # Prefer OpenF1 starting_grid. Fallback only for grid, never for results/labels.
        position_clean_path = os.path.join(CLEAN_DATA_DIR, "position.csv")
        starting_grid_clean_path = os.path.join(CLEAN_DATA_DIR, "starting_grid.csv")

        if os.path.exists(position_clean_path):
            pos = pd.read_csv(position_clean_path, low_memory=False)
            if "date" in pos.columns:
                pos = pos.sort_values("date")
            if {"session_key", "driver_number"}.issubset(pos.columns):
                if not os.path.exists(starting_grid_clean_path):
                    grid = pos.groupby(["session_key", "driver_number"], as_index=False).first()
                    grid.to_csv(starting_grid_clean_path, index=False)
                    logger.warning(
                        "starting_grid.csv derived from position.csv (FALLBACK). "
                        "Position telemetry may not represent the real starting grid exactly."
                    )
                else:
                    logger.info("starting_grid.csv exists from OpenF1; skipping position fallback")

    print("\n" + "=" * 60)
    print(f"{'File':<25} {'Raw':>10} {'Cleaned':>10} {'Dropped':>10}  Status")
    print("-" * 60)
    for s in summary:
        print(
            f"{s['file']:<25} "
            f"{str(s.get('raw_rows', '-')):>10} "
            f"{str(s.get('clean_rows', '-')):>10} "
            f"{str(s.get('dropped', '-')):>10}  "
            f"{s['status']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    clean_all()

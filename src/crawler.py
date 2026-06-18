

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import fastf1
import pandas as pd
import requests

from .config import load_config, PipelineConfig
from .utils import setup_logging, utc_now_iso, write_json

logger = logging.getLogger(__name__)

# ─── Endpoint definitions ─────────────────────────────────────────────────────
# Telemetry: lưu Parquet vì dữ liệu mili-giây, rất lớn
TELEMETRY_ENDPOINTS = {"car_data", "location"}

# Tất cả endpoints thông thường (dữ liệu bảng, lưu CSV)
NORMAL_ENDPOINTS = [
    "meetings", "drivers", "laps", "stints", "position",
    "intervals", "pit", "starting_grid", "race_control",
    "weather", "session_result", "team_radio", "overtakes",
]

# Endpoints theo năm (không theo session_key)
PER_YEAR_ENDPOINTS = {"championship_drivers", "championship_teams"}


# ─── Rate Limiter (Thread-safe) ───────────────────────────────────────────────
class RateLimiter:
    """
    Đảm bảo tổng số request trong khoảng `period` giây không vượt `max_calls`.
    Dùng cho cả môi trường đa luồng — có lock để tránh race condition.
    """

    def __init__(self, max_calls: int = 28, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            # Loại bỏ timestamp cũ hơn `period` giây
            self._calls = [t for t in self._calls if now - t < self.period]
            if len(self._calls) >= self.max_calls:
                sleep_time = self.period - (now - self._calls[0]) + 0.1
                if sleep_time > 0:
                    logger.info(
                        f"[RateLimiter] Đạt {self.max_calls} req/{self.period:.0f}s. "
                        f"Chờ {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
            self._calls.append(time.time())


# Khởi tạo rate limiter toàn cục (dùng chung giữa các thread)
_rate_limiter = RateLimiter(max_calls=28, period=60.0)


# ─── Skip Existing ────────────────────────────────────────────────────────────
def _file_exists(raw_dir: Path, endpoint: str, year: int, session_key: int) -> bool:
    """Trả về True nếu file raw của endpoint này đã được tải về."""
    dir_path = raw_dir / endpoint / str(year)
    for ext in ("csv", "parquet", "json"):
        if (dir_path / f"{session_key}.{ext}").exists():
            return True
    return False


# ─── Fetch Functions ──────────────────────────────────────────────────────────
def _fetch_api(
    base_url: str, endpoint: str, params: dict, retries: int = 3
) -> Optional[List[dict]]:
    """
    Gọi OpenF1 REST API với retry + rate limiting.
    Trả về list[dict] hoặc None nếu thất bại.
    """
    url = f"{base_url}/{endpoint}"
    for attempt in range(retries):
        try:
            _rate_limiter.wait()
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 429:
                wait = (attempt + 1) * 10
                logger.warning(f"  [429] Rate limited on {endpoint}. Chờ {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            return data if data else None
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            logger.error(f"  [FAIL] {endpoint}: {exc}")
            return None
    return None


def _fetch_telemetry_fastf1(
    year: int,
    meeting_name: str,
    session_name: str,
    endpoint: str,
    cache_dir: Path,
    session_key: int,
) -> Optional[List[dict]]:
    """
    Tải telemetry (car_data / location) từ FastF1.

    Sử dụng FastF1 thay vì OpenF1 REST vì:
      - Nhanh hơn (dữ liệu đã được nén sẵn)
      - Có lap_number và compound đính kèm từng dòng
        → Giúp aggregate_telemetry_to_lap() hoạt động chính xác
      - Độ tin cậy cao hơn (FastF1 tự xử lý retry)
    """
    try:
        fastf1.Cache.enable_cache(str(cache_dir))
        logger.info(f"  [FastF1] Loading: {year} | {meeting_name} | {session_name}")
        session = fastf1.get_session(year, meeting_name, session_name)
        session.load(telemetry=True, laps=True, weather=False)

        all_data: list[dict] = []
        for driver in session.drivers:
            try:
                drv_laps = session.laps.pick_driver(driver)
                for _, lap in drv_laps.iterlaps():
                    if endpoint == "car_data":
                        tele = lap.get_telemetry()
                    else:  # location
                        tele = lap.get_pos_data()

                    if not tele.empty:
                        records = tele.to_dict("records")
                        for r in records:
                            r["driver_number"]  = driver
                            r["session_key"]    = session_key
                            r["lap_number"]     = lap["LapNumber"]
                            r["compound"]       = lap["Compound"]
                        all_data.extend(records)
            except Exception as exc:
                logger.debug(f"    Driver {driver}: {exc}")

        return all_data if all_data else None
    except Exception as exc:
        logger.warning(f"  [FastF1] Error: {exc}")
        return None


# ─── Save Function ────────────────────────────────────────────────────────────
def _save(
    data: List[dict],
    endpoint: str,
    year: int,
    session_key: int,
    raw_dir: Path,
    dry_run: bool = False,
) -> None:
    """Lưu dữ liệu đã tải về disk. Telemetry → Parquet (zstd), còn lại → CSV."""
    if dry_run:
        logger.info(f"  [DryRun] Would save {len(data):,} rows → {endpoint}/{year}/{session_key}")
        return

    is_tele = endpoint in TELEMETRY_ENDPOINTS
    ext = "parquet" if is_tele else "csv"
    path = raw_dir / endpoint / str(year) / f"{session_key}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.DataFrame(data)
        if is_tele:
            df.to_parquet(path, index=False, compression="zstd")
        else:
            df.to_csv(path, index=False)
        logger.info(f"  ✓ {endpoint}/{year}/{session_key}.{ext}  ({len(data):,} rows)")
    except Exception as exc:
        logger.error(f"  ✗ Save failed {path}: {exc}")


# ─── Per-Endpoint Worker (dùng trong ThreadPoolExecutor) ─────────────────────
def _fetch_and_save_one(
    endpoint: str,
    session: dict,
    year: int,
    config: PipelineConfig,
    dry_run: bool,
) -> tuple[str, bool]:
    """
    Tải + lưu một endpoint cho một session.
    Thread-safe: mỗi luồng xử lý một endpoint độc lập.
    """
    session_key  = session["session_key"]
    session_name = session.get("session_name", "")
    meeting_name = session.get("meeting_name", "")

    # Skip nếu file đã tồn tại
    if _file_exists(config.raw_dir, endpoint, year, session_key):
        logger.info(f"  [SKIP] {endpoint} [{session_key}] — already exists")
        return endpoint, True

    # Xác định params
    if endpoint == "meetings":
        meeting_key = session.get("meeting_key")
        params = {"meeting_key": meeting_key} if meeting_key else {"year": year}
    elif endpoint in PER_YEAR_ENDPOINTS:
        params = {"year": year}
    else:
        params = {"session_key": session_key}

    data = _fetch_api(config.base_url, endpoint, params)
    if data:
        _save(data, endpoint, year, session_key, config.raw_dir, dry_run)
        return endpoint, True
    else:
        logger.warning(f"  [NO DATA] {endpoint} [{session_key}]")
        return endpoint, False


# ─── Main Orchestrator ────────────────────────────────────────────────────────
def run_crawler(
    config_path: str | None = None,
    dry_run: bool = False,
    max_workers: int = 4,
) -> dict:
    """
    Crawl toàn bộ dữ liệu F1 theo cấu hình trong pipeline_config.yaml.

    Args:
        config_path : Đường dẫn đến file YAML. None = dùng mặc định.
        dry_run     : Nếu True, không lưu file, chỉ in ra những gì sẽ làm.
        max_workers : Số luồng chạy song song cho các endpoint thường.
                      Telemetry luôn chạy tuần tự (dữ liệu quá nặng).

    Returns:
        dict chứa metadata của lần chạy (năm, số session, trạng thái).
    """
    config = load_config(config_path)
    setup_logging()

    cache_dir = config.project_root / "data" / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    if dry_run:
        logger.warning("!!! DRY-RUN MODE: Không có file nào được lưu !!!")

    total_sessions = 0
    yearly_meta = []

    for year in range(config.start_year, config.end_year + 1):
        logger.info(f"\n{'═'*60}")
        logger.info(f"  Năm {year}")
        logger.info(f"{'═'*60}")

        # Lấy danh sách sessions
        sessions = _fetch_api(config.base_url, "sessions", {"year": year})
        if not sessions:
            logger.warning(f"Không có session nào cho năm {year}. Bỏ qua.")
            continue

        # Lọc theo session_type trong config
        if config.session_types:
            sessions = [s for s in sessions if s.get("session_name") in config.session_types]

        # Giới hạn số session mỗi năm (config: limit_sessions)
        limit = config.limit_sessions
        sessions = sessions[:limit]

        logger.info(f"  Sẽ xử lý {len(sessions)} session(s) (limit={limit})")
        total_sessions += len(sessions)

        for session in sessions:
            session_key  = session.get("session_key")
            session_name = session.get("session_name", "?")
            meeting_name = session.get("meeting_name", "?")

            logger.info(f"\n  ─── {session_name} | {meeting_name} [key={session_key}] ───")

            # ── Lưu metadata session ─────────────────────────────────────────
            if not _file_exists(config.raw_dir, "sessions", year, session_key):
                _save([session], "sessions", year, session_key, config.raw_dir, dry_run)
            else:
                logger.info(f"  [SKIP] sessions [{session_key}] — already exists")

            # ── Các endpoint thường: chạy SONG SONG ─────────────────────────
            logger.info(f"  [Parallel] {len(NORMAL_ENDPOINTS)} endpoints × {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_fetch_and_save_one, ep, session, year, config, dry_run): ep
                    for ep in NORMAL_ENDPOINTS
                }
                for future in as_completed(futures):
                    ep = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(f"  ✗ {ep} raised: {exc}")

            # ── Telemetry (car_data + location): chạy TUẦN TỰ ───────────────
            logger.info("  [Sequential] Telemetry endpoints (FastF1)...")
            for ep in TELEMETRY_ENDPOINTS:
                if _file_exists(config.raw_dir, ep, year, session_key):
                    logger.info(f"  [SKIP] {ep} [{session_key}] — already exists")
                    continue

                data = _fetch_telemetry_fastf1(
                    year, meeting_name, session_name, ep, cache_dir, session_key
                )
                if data:
                    _save(data, ep, year, session_key, config.raw_dir, dry_run)
                else:
                    logger.warning(f"  [NO DATA] {ep} [{session_key}]")

            logger.info(f"  ✅ Xong: {meeting_name}")

        yearly_meta.append({"year": year, "session_count": len(sessions)})

    # ── Lưu metadata ──────────────────────────────────────────────────────────
    metadata = {
        "crawled_at":     utc_now_iso(),
        "years":          list(range(config.start_year, config.end_year + 1)),
        "total_sessions": total_sessions,
        "limit_sessions": config.limit_sessions,
        "dry_run":        dry_run,
        "status":         "ok",
        "per_year":       yearly_meta,
    }
    if not dry_run:
        write_json(metadata, config.metadata_dir / "crawler_metadata.json")

    logger.info(f"\n[Crawler] Hoàn tất! Tổng {total_sessions} session(s) đã xử lý.")
    return metadata


if __name__ == "__main__":
    run_crawler()

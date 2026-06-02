"""
src/crawler.py
────────────────────────────────────────────────────────────────────────────
Unified F1 Data Crawler — Bronze Layer
────────────────────────────────────────────────────────────────────────────
Module crawl chính thức và duy nhất của project. Tích hợp đầy đủ:
  - Đọc cấu hình từ configs/pipeline_config.yaml
  - Skip Existing: bỏ qua file đã tải, tránh crawl lại từ đầu
  - Multithreading: tải nhiều endpoint song song (ThreadPoolExecutor)
  - Rate Limiter: tự động chờ khi gần chạm giới hạn 30 req/phút của OpenF1
  - FastF1 Telemetry: tải car_data/location chất lượng cao với lap_number + compound

Kiến trúc:
  run_crawler()               ← Hàm chính, được gọi từ run_e2e_pipeline.py
    └── _process_session()    ← Xử lý một session
          ├── _fetch_normal_endpoints()  ← Chạy song song (ThreadPoolExecutor)
          └── _fetch_telemetry()         ← Chạy tuần tự (dữ liệu nặng)
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
DEFAULT_TELEMETRY_ENDPOINTS = {"car_data", "location"}

# Tất cả endpoints thông thường (dữ liệu bảng, lưu CSV)
NORMAL_ENDPOINTS = []

# Endpoints theo năm (không theo session_key)
PER_YEAR_ENDPOINTS = {"meetings", "championship_drivers", "championship_teams"}


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


# ─── Skip Existing ────────────────────────────────────────────────────────────
def _file_exists(raw_dir: Path, endpoint: str, year: int, session_key: int) -> bool:
    """Trả về True nếu file raw của endpoint này đã được tải về."""
    dir_path = raw_dir / endpoint / str(year)
    for ext in ("csv", "parquet", "json"):
        if (dir_path / f"{session_key}.{ext}").exists():
            return True
    return False


def _should_skip_existing(config: PipelineConfig, endpoint: str, year: int, session_key: int) -> bool:
    return config.resume_existing_files and _file_exists(config.raw_dir, endpoint, year, session_key)


def _should_skip_existing_year(config: PipelineConfig, endpoint: str, year: int) -> bool:
    return config.resume_existing_files and _year_file_exists(config.raw_dir, endpoint, year)


def _reset_raw_dir_if_configured(config: PipelineConfig, dry_run: bool) -> None:
    if not config.delete_raw_before_crawl:
        return

    preserve = set(config.raw_reset_preserve_endpoints)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    for endpoint_dir in config.raw_dir.iterdir():
        if not endpoint_dir.is_dir() or endpoint_dir.name in preserve:
            continue
        if dry_run:
            logger.info("[DryRun] Would delete raw endpoint directory: %s", endpoint_dir)
        else:
            shutil.rmtree(endpoint_dir)
            logger.info("Deleted raw endpoint directory before crawl: %s", endpoint_dir)


def _write_crawler_checkpoint(config: PipelineConfig, payload: dict) -> None:
    if config.crawler_checkpoint_enabled:
        write_json(payload, config.crawler_checkpoint_path)


def _year_file_exists(raw_dir: Path, endpoint: str, year: int) -> bool:
    dir_path = raw_dir / endpoint
    for ext in ("csv", "parquet", "json"):
        if (dir_path / str(year) / f"{endpoint}.{ext}").exists():
            return True
        if (dir_path / f"{year}.{ext}").exists():
            return True
        if (dir_path / str(year) / f"{year}.{ext}").exists():
            return True
    return False


def _filter_completed_sessions(
    sessions: list[dict],
    include_future_sessions: bool,
    completed_before_utc: str | None = None,
) -> list[dict]:
    if include_future_sessions:
        return sessions
    now = (
        pd.to_datetime(completed_before_utc, errors="coerce", utc=True)
        if completed_before_utc
        else pd.Timestamp(datetime.now(timezone.utc))
    )
    if pd.isna(now):
        now = pd.Timestamp(datetime.now(timezone.utc))
    completed = []
    for session in sessions:
        date_start = pd.to_datetime(session.get("date_start"), errors="coerce", utc=True)
        if pd.notna(date_start) and date_start <= now:
            completed.append(session)
    return completed


# ─── Fetch Functions ──────────────────────────────────────────────────────────
def _fetch_api(
    config: PipelineConfig,
    endpoint: str,
    params: dict,
    rate_limiter: RateLimiter,
) -> Optional[List[dict]]:
    """
    Gọi OpenF1 REST API với retry + rate limiting.
    Trả về list[dict] hoặc None nếu thất bại.
    """
    url = f"{config.base_url}/{endpoint}"
    retries = config.max_retries
    for attempt in range(retries):
        try:
            rate_limiter.wait()
            response = requests.get(url, params=params, timeout=config.timeout_seconds)
            if config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)
            if response.status_code == 429:
                wait = config.retry_wait_seconds
                logger.warning(f"  [429] Rate limited on {endpoint}. Chờ {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            return data if data else None
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(config.retry_wait_seconds)
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
    config: PipelineConfig,
    dry_run: bool = False,
) -> None:
    """Lưu dữ liệu đã tải về disk. Telemetry → Parquet (zstd), còn lại → CSV."""
    if dry_run:
        logger.info(f"  [DryRun] Would save {len(data):,} rows → {endpoint}/{year}/{session_key}")
        return

    is_tele = endpoint in DEFAULT_TELEMETRY_ENDPOINTS
    ext = config.telemetry_format if is_tele else "csv"
    if is_tele and ext != "parquet":
        raise ValueError(f"Unsupported telemetry_format={ext!r}; only parquet is currently supported.")
    path = config.raw_dir / endpoint / str(year) / f"{session_key}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.DataFrame(data)
        if is_tele:
            df.to_parquet(path, index=config.csv_index, compression=config.telemetry_compression)
        else:
            df.to_csv(path, index=config.csv_index, encoding=config.csv_encoding)
        logger.info(f"  ✓ {endpoint}/{year}/{session_key}.{ext}  ({len(data):,} rows)")
    except Exception as exc:
        logger.error(f"  ✗ Save failed {path}: {exc}")


# ─── Per-Endpoint Worker (dùng trong ThreadPoolExecutor) ─────────────────────
def _save_year(
    data: List[dict],
    endpoint: str,
    year: int,
    config: PipelineConfig,
    dry_run: bool = False,
) -> None:
    if dry_run:
        logger.info(f"  [DryRun] Would save {len(data):,} rows -> {endpoint}/{year}/{endpoint}.csv")
        return

    path = config.raw_dir / endpoint / str(year) / f"{endpoint}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(data).to_csv(path, index=config.csv_index, encoding=config.csv_encoding)
        logger.info(f"  Saved {endpoint}/{year}/{endpoint}.csv ({len(data):,} rows)")
    except Exception as exc:
        logger.error(f"  Save failed {path}: {exc}")


def _fetch_and_save_one(
    endpoint: str,
    session: dict,
    year: int,
    config: PipelineConfig,
    rate_limiter: RateLimiter,
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
    if _should_skip_existing(config, endpoint, year, session_key):
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

    data = _fetch_api(config, endpoint, params, rate_limiter)
    if data:
        _save(data, endpoint, year, session_key, config, dry_run)
        return endpoint, True
    else:
        logger.warning(f"  [NO DATA] {endpoint} [{session_key}]")
        return endpoint, False


# ─── Main Orchestrator ────────────────────────────────────────────────────────
def run_crawler(
    config_path: str | None = None,
    dry_run: bool | None = None,
    max_workers: int | None = None,
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
    setup_logging(
        level=config.log_level,
        log_file_path=config.log_file_path,
        log_to_file=config.log_to_file,
        retention_days=config.log_retention_days,
    )
    dry_run = config.dry_run_default if dry_run is None else dry_run
    max_workers = config.crawler_max_workers if max_workers is None else max_workers

    if config.freeze_existing_raw:
        logger.info(
            "Bronze snapshot is frozen; crawl step will not fetch new raw files. "
            "Set sessions.freeze_existing_raw=false to backfill from OpenF1/FastF1."
        )
        metadata = {
            "crawled_at": utc_now_iso(),
            "years": list(range(config.start_year, config.end_year + 1)),
            "total_sessions": 0,
            "limit_sessions": config.limit_sessions,
            "dry_run": dry_run,
            "status": "frozen_existing_raw",
        }
        if not dry_run:
            write_json(metadata, config.metadata_dir / "crawl_metadata.json")
        return metadata

    _reset_raw_dir_if_configured(config, dry_run)

    cache_dir = config.fastf1_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    rate_limiter = RateLimiter(
        max_calls=config.rate_limit_max_calls,
        period=config.rate_limit_period_seconds,
    )
    logger.info(
        "API rate limit from config: %s request(s) per %.1f second(s); post-request sleep %.2fs",
        config.rate_limit_max_calls,
        config.rate_limit_period_seconds,
        config.sleep_seconds,
    )

    if dry_run:
        logger.warning("!!! DRY-RUN MODE: Không có file nào được lưu !!!")

    telemetry_endpoints = set(config.telemetry_endpoints or DEFAULT_TELEMETRY_ENDPOINTS)
    normal_endpoints = [
        endpoint for endpoint in config.per_session_endpoints
        if endpoint != "sessions"
        and endpoint not in config.per_year_endpoints
        and endpoint not in telemetry_endpoints
    ]

    total_sessions = 0
    yearly_meta = []

    for year in range(config.start_year, config.end_year + 1):
        logger.info(f"\n{'═'*60}")
        logger.info(f"  Năm {year}")
        logger.info(f"{'═'*60}")

        # Lấy danh sách sessions
        sessions = _fetch_api(config, "sessions", {"year": year}, rate_limiter)
        if not sessions:
            logger.warning(f"Không có session nào cho năm {year}. Bỏ qua.")
            continue

        # Lọc theo session_type trong config
        if config.session_types:
            sessions = [s for s in sessions if s.get("session_name") in config.session_types]

        before_future_filter = len(sessions)
        sessions = _filter_completed_sessions(
            sessions,
            config.include_future_sessions,
            config.completed_before_utc,
        )
        if len(sessions) != before_future_filter:
            logger.info(
                "  Filtered out %s session(s) after completion cutoff because include_future_sessions=false",
                before_future_filter - len(sessions),
            )

        # Giới hạn số session mỗi năm (config: limit_sessions)
        for endpoint in config.per_year_endpoints:
            if _should_skip_existing_year(config, endpoint, year):
                logger.info(f"  [SKIP] {endpoint}/{year}.csv already exists")
                continue
            data = _fetch_api(config, endpoint, {"year": year}, rate_limiter)
            if data:
                _save_year(data, endpoint, year, config, dry_run)
            else:
                logger.warning(f"  [NO DATA] {endpoint} [{year}]")

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
            if not _should_skip_existing(config, "sessions", year, session_key):
                _save([session], "sessions", year, session_key, config, dry_run)
            else:
                logger.info(f"  [SKIP] sessions [{session_key}] — already exists")

            # ── Các endpoint thường: chạy SONG SONG ─────────────────────────
            logger.info(f"  [Parallel] {len(normal_endpoints)} endpoints x {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_fetch_and_save_one, ep, session, year, config, rate_limiter, dry_run): ep
                    for ep in normal_endpoints
                }
                for future in as_completed(futures):
                    ep = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(f"  ✗ {ep} raised: {exc}")

            # ── Telemetry (car_data + location): chạy TUẦN TỰ ───────────────
            logger.info("  [Sequential] Telemetry endpoints (FastF1)...")
            for ep in (telemetry_endpoints if config.telemetry_enabled else []):
                if _should_skip_existing(config, ep, year, session_key):
                    logger.info(f"  [SKIP] {ep} [{session_key}] — already exists")
                    continue

                if not config.telemetry_crawl_if_missing:
                    logger.info(f"  [SKIP] {ep} [{session_key}] missing locally; telemetry crawl disabled")
                    continue

                data = _fetch_telemetry_fastf1(
                    year, meeting_name, session_name, ep, cache_dir, session_key
                )
                if data:
                    _save(data, ep, year, session_key, config, dry_run)
                else:
                    logger.warning(f"  [NO DATA] {ep} [{session_key}]")

            logger.info(f"  ✅ Xong: {meeting_name}")
            _write_crawler_checkpoint(config, {
                "updated_at": utc_now_iso(),
                "year": year,
                "session_key": session_key,
                "session_name": session_name,
                "meeting_name": meeting_name,
                "status": "session_processed",
            })

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

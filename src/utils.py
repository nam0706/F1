from __future__ import annotations

import json
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(
    level: int | str = logging.INFO,
    log_file_path: Path | str | None = None,
    log_to_file: bool = False,
    retention_days: int = 14,
) -> None:
    resolved_level = getattr(logging, str(level).upper(), level)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_to_file and log_file_path is not None:
        path = Path(log_file_path)
        ensure_dir(path.parent)
        file_handler = TimedRotatingFileHandler(
            path,
            when="D",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=resolved_level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8", **kwargs)


def write_markdown(content: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

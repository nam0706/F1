from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime with clear message
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.yaml"


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def steps_to_run(self) -> list[str]:
        # Mặc định chạy tất cả nếu không có config
        exec_cfg = self.raw.get("execution", {})
        return list(exec_cfg.get("steps_to_run", ["crawl", "clean", "validate", "feature"]))

    @property
    def execution_mode(self) -> str:
        return str(self.raw.get("execution", {}).get("mode", "full"))

    @property
    def dry_run_default(self) -> bool:
        return bool(self.raw.get("execution", {}).get("dry_run", False))

    @property
    def sample_mode(self) -> bool:
        return bool(self.raw.get("execution", {}).get("sample_mode", False))

    @property
    def sample_session_limit(self) -> int | None:
        value = self.raw.get("execution", {}).get("sample_session_limit")
        if value is None or str(value).strip().lower() in ("", "none", "null", "~"):
            return None
        return int(value)

    @property
    def start_year(self) -> int:
        return int(self.raw["data"]["start_year"])

    @property
    def end_year(self) -> int:
        return int(self.raw["data"]["end_year"])

    @property
    def base_url(self) -> str:
        return str(self.raw["api"]["base_url"]).rstrip("/")

    @property
    def session_types(self) -> list[str]:
        return list(self.raw["sessions"]["types"])

    @property
    def include_future_sessions(self) -> bool:
        return bool(self.raw["sessions"].get("include_future_sessions", True))

    @property
    def completed_before_utc(self) -> str | None:
        value = self.raw["sessions"].get("completed_before_utc")
        if value is None or str(value).strip().lower() in ("", "none", "null", "~"):
            return None
        return str(value)

    @property
    def freeze_existing_raw(self) -> bool:
        return bool(self.raw["sessions"].get("freeze_existing_raw", False))

    @property
    def limit_sessions(self) -> int:
        return int(self.raw["sessions"].get("limit_sessions", 99))

    @property
    def per_year_endpoints(self) -> list[str]:
        return list(self.raw["raw_endpoints"].get("per_year", []))

    @property
    def per_session_endpoints(self) -> list[str]:
        return list(self.raw["raw_endpoints"]["per_session"])

    @property
    def preserve_raw_endpoints(self) -> list[str]:
        return list(self.raw.get("preserve_raw_endpoints", []))

    def data_path(self, key: str) -> Path:
        value = self.raw["data"][key]
        return self.project_root / value

    @property
    def raw_dir(self) -> Path:
        return self.data_path("raw_dir")

    @property
    def cleaned_dir(self) -> Path:
        return self.data_path("cleaned_dir")

    @property
    def features_dir(self) -> Path:
        return self.project_root / str(self.raw["data"].get("features_dir", "data/gold/features"))

    @property
    def processed_dir(self) -> Path:
        return self.project_root / str(self.raw["data"].get("processed_dir", "data/gold"))

    @property
    def gold_dir(self) -> Path:
        return self.project_root / str(self.raw["data"].get("gold_dir", "data/gold"))

    @property
    def metadata_dir(self) -> Path:
        return self.data_path("metadata_dir")

    @property
    def reports_dir(self) -> Path:
        return self.data_path("reports_dir")

    @property
    def logging_config(self) -> dict[str, Any]:
        return dict(self.raw.get("logging", {}))

    @property
    def log_level(self) -> str:
        return str(self.logging_config.get("level", "INFO")).upper()

    @property
    def log_to_file(self) -> bool:
        return bool(self.logging_config.get("log_to_file", False))

    @property
    def log_file_path(self) -> Path:
        return self.project_root / str(self.logging_config.get("log_file_path", "reports/logs/pipeline.log"))

    @property
    def log_retention_days(self) -> int:
        return int(self.logging_config.get("retention_days", 14))

    @property
    def sleep_seconds(self) -> float:
        return float(self.raw["api"]["sleep_seconds"])

    @property
    def rate_limit_max_calls(self) -> int:
        return int(self.raw["api"].get("rate_limit_max_calls", 28))

    @property
    def rate_limit_period_seconds(self) -> float:
        return float(self.raw["api"].get("rate_limit_period_seconds", 60.0))

    @property
    def retry_wait_seconds(self) -> float:
        return float(self.raw["api"]["retry_wait_seconds"])

    @property
    def max_retries(self) -> int:
        return int(self.raw["api"]["max_retries"])

    @property
    def timeout_seconds(self) -> int:
        return int(self.raw["api"]["timeout_seconds"])

    @property
    def crawler_config(self) -> dict[str, Any]:
        return dict(self.raw.get("crawler", {}))

    @property
    def crawler_max_workers(self) -> int:
        return int(self.crawler_config.get("max_workers", 4))

    @property
    def fastf1_cache_dir(self) -> Path:
        return self.project_root / str(self.crawler_config.get("fastf1_cache_dir", ".cache/fastf1"))

    @property
    def resume_existing_files(self) -> bool:
        return bool(self.crawler_config.get("resume_existing_files", True))

    @property
    def crawler_checkpoint_config(self) -> dict[str, Any]:
        return dict(self.crawler_config.get("checkpoint", {}))

    @property
    def crawler_checkpoint_enabled(self) -> bool:
        return bool(self.crawler_checkpoint_config.get("enabled", False))

    @property
    def crawler_checkpoint_path(self) -> Path:
        return self.project_root / str(self.crawler_checkpoint_config.get("path", "data/metadata/crawler_checkpoint.json"))

    @property
    def raw_reset_policy(self) -> dict[str, Any]:
        return dict(self.crawler_config.get("raw_reset_policy", {}))

    @property
    def delete_raw_before_crawl(self) -> bool:
        return bool(self.raw_reset_policy.get("delete_raw_before_crawl", False))

    @property
    def raw_reset_preserve_endpoints(self) -> list[str]:
        configured = self.raw_reset_policy.get("preserve_endpoints")
        if configured is not None:
            return list(configured)
        return self.preserve_raw_endpoints

    @property
    def storage_config(self) -> dict[str, Any]:
        return dict(self.raw.get("storage", {}))

    @property
    def csv_encoding(self) -> str:
        return str(self.storage_config.get("csv_encoding", "utf-8"))

    @property
    def csv_index(self) -> bool:
        return bool(self.storage_config.get("csv_index", False))

    @property
    def telemetry_format(self) -> str:
        return str(self.storage_config.get("telemetry_format", "parquet")).lower()

    @property
    def telemetry_compression(self) -> str | None:
        value = self.storage_config.get("telemetry_compression", "zstd")
        if value is None or str(value).strip().lower() in ("", "none", "null", "~"):
            return None
        return str(value)

    @property
    def silver_large_format(self) -> str:
        return str(self.storage_config.get("silver_large_format", "parquet")).lower()

    @property
    def silver_large_compression(self) -> str | None:
        value = self.storage_config.get("silver_large_compression", "zstd")
        if value is None or str(value).strip().lower() in ("", "none", "null", "~"):
            return None
        return str(value)

    @property
    def gold_format(self) -> str:
        return str(self.storage_config.get("gold_format", "parquet")).lower()

    @property
    def gold_compression(self) -> str | None:
        value = self.storage_config.get("gold_compression", "zstd")
        if value is None or str(value).strip().lower() in ("", "none", "null", "~"):
            return None
        return str(value)

    @property
    def quality_config(self) -> dict[str, Any]:
        return dict(self.raw.get("quality", {}))

    @property
    def partition_contract(self) -> dict[str, Any]:
        return dict(self.raw.get("partition_contract", {}))

    @property
    def schema_contract(self) -> dict[str, Any]:
        return dict(self.raw.get("schema_contract", {}))

    @property
    def telemetry_enabled(self) -> bool:
        return bool(self.raw.get("telemetry", {}).get("enabled", False))

    @property
    def telemetry_endpoints(self) -> list[str]:
        return list(self.raw.get("telemetry", {}).get("endpoints", ["car_data", "location"]))

    @property
    def telemetry_crawl_if_missing(self) -> bool:
        return bool(self.raw.get("telemetry", {}).get("crawl_if_missing", True))

    @property
    def telemetry_aggregate(self) -> bool:
        return bool(self.raw.get("telemetry", {}).get("aggregate", False))

    @property
    def telemetry_join_to_master(self) -> bool:
        return bool(self.raw.get("telemetry", {}).get("join_to_master", False))

    @property
    def telemetry_keep_raw(self) -> bool:
        return bool(self.raw.get("telemetry", {}).get("keep_raw", True))

    @property
    def telemetry_aggregate_cache_path(self) -> Path:
        value = self.raw.get("telemetry", {}).get(
            "aggregate_cache_path",
            "data/metadata/telemetry_lap_features.parquet",
        )
        return self.project_root / value

    @property
    def resample_interval(self) -> str | None:
        """Khoảng resample cho telemetry. None = không resample (giữ nguyên raw)."""
        value = self.raw.get("telemetry", {}).get("resample_interval")
        if value is None or str(value).strip().lower() in ("none", "null", "~", ""):
            return None
        return str(value)


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs. Install it with: pip install pyyaml")

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(raw=raw, path=path)

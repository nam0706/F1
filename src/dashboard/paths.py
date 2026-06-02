from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DashboardPaths:
    project_root: Path = PROJECT_ROOT

    @property
    def training_path(self) -> Path:
        return self.project_root / "data" / "gold" / "training" / "finish_bucket_live_any_lap.parquet"

    @property
    def leakage_path(self) -> Path:
        return self.project_root / "data" / "gold" / "metadata" / "leakage_report.json"

    @property
    def feature_contract_path(self) -> Path:
        return self.project_root / "data" / "gold" / "metadata" / "feature_contract.json"

    @property
    def training_contract_path(self) -> Path:
        return self.project_root / "data" / "gold" / "metadata" / "training_dataset_contract.json"

    @property
    def model_metadata_path(self) -> Path:
        return self.project_root / "models" / "gold_model_metadata.json"

    @property
    def model_path(self) -> Path:
        return self.project_root / "models" / "gold_finish_bucket_model.joblib"

    @property
    def required_artifacts(self) -> list[Path]:
        return [
            self.training_path,
            self.model_path,
            self.model_metadata_path,
            self.leakage_path,
            self.feature_contract_path,
            self.training_contract_path,
        ]


PATHS = DashboardPaths()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldPaths:
    root: Path

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    def ensure(self) -> None:
        for path in [self.labels_dir, self.features_dir, self.training_dir, self.metadata_dir]:
            path.mkdir(parents=True, exist_ok=True)


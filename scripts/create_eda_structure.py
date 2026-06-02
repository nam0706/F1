"""Create the EDA workspace structure for the F1 WinRate Predictor project."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


STRUCTURE: dict[str, list[str]] = {
    "eda/shared/scripts": ["config.py", "file_utils.py", "validation_utils.py", "viz_utils.py"],
    "eda/shared/logs": ["execution_log.txt"],
    "eda/bronze/notebooks": [
        "01_file_integrity.ipynb",
        "02_schema_validation.ipynb",
        "03_pk_fk_checks.ipynb",
        "04_null_analysis.ipynb",
        "05_range_validation.ipynb",
    ],
    "eda/bronze/outputs/tables": [],
    "eda/bronze/outputs/charts": [],
    "eda/bronze/outputs/reports": [],
    "eda/bronze/insights": ["01_summary.md", "02_issues_found.md", "03_next_strategy.md"],
    "eda/bronze/checkpoints": ["bronze_completed.txt"],
    "eda/silver/notebooks": ["01_cleaning_log.ipynb", "02_missing_value_handling.ipynb", "03_outlier_treatment.ipynb"],
    "eda/silver/outputs/tables": [],
    "eda/silver/outputs/charts": [],
    "eda/silver/outputs/reports": [],
    "eda/silver/insights": ["01_cleaning_summary.md", "02_quality_assessment.md", "03_ready_for_feature_engineering.md"],
    "eda/silver/checkpoints": ["silver_completed.txt"],
    "eda/gold/notebooks": ["01_lap_time_features.ipynb", "02_tyre_performance.ipynb", "03_overtake_analysis.ipynb"],
    "eda/gold/outputs/tables": [],
    "eda/gold/outputs/charts": [],
    "eda/gold/outputs/reports": [],
    "eda/gold/insights": ["01_feature_selection.md", "02_model_preparation.md"],
    "eda/gold/checkpoints": ["gold_completed.txt"],
}


README_TEXT = """# {title}

## Purpose

This directory is part of the layered EDA workspace for the F1 WinRate Predictor project.

## Usage

- `notebooks/`: executable analysis assets.
- `outputs/`: generated tables, charts, and reports.
- `insights/`: human-readable findings and next steps.
- `checkpoints/`: completion markers for review.
"""


def write_if_missing(path: Path, content: str = "") -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def make_placeholder_notebook(path: Path, title: str) -> None:
    if path.exists():
        return
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(f"# {title}\n\nThis notebook is a placeholder for the EDA workflow."),
        nbformat.v4.new_code_cell("from pathlib import Path\n\nROOT = Path.cwd()\nROOT"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


def create_structure() -> None:
    for folder, files in STRUCTURE.items():
        folder_path = ROOT / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        write_if_missing(folder_path / "README.md", README_TEXT.format(title=folder))

        for filename in files:
            path = folder_path / filename
            if filename.endswith(".ipynb"):
                make_placeholder_notebook(path, filename.replace("_", " ").replace(".ipynb", "").title())
            elif filename.endswith(".md"):
                write_if_missing(
                    path,
                    f"# {filename.replace('.md', '').replace('_', ' ').title()}\n\n## Key Findings\n\n- TBD\n\n## Recommendations\n\n- TBD\n",
                )
            elif filename.endswith(".txt"):
                write_if_missing(path, "Status: pending\n")
            elif filename.endswith(".py"):
                write_if_missing(path, "")
            else:
                write_if_missing(path, "")


if __name__ == "__main__":
    create_structure()
    print("EDA structure created successfully.")

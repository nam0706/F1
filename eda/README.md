# EDA Workspace

## Overview

This directory contains Exploratory Data Analysis assets organized by medallion layer:

- `bronze/`: raw source-layer profiling and integrity checks.
- `silver/`: cleaned dataset quality and transformation review.
- `gold/`: feature engineering and model-readiness analysis.
- `shared/`: reusable scripts, configuration access, logs, and utilities.

## How to Review

1. Read `*/insights/` first for findings and decisions.
2. Open `*/outputs/tables/` for CSV evidence.
3. Open `*/outputs/charts/` for exported visuals.
4. Review `*/notebooks/` only when code-level inspection is needed.

## Execution

Run notebooks in order within each layer. The Bronze layer should complete before Silver, and Silver should complete before Gold.

```powershell
cd D:\F1_WinRate_Predictor
.\venv\Scripts\jupyter.exe notebook eda\bronze\notebooks\01_file_integrity.ipynb
```

## Validation Levels

| Layer | Focus | Critical Outcome |
|---|---|---|
| Bronze | File integrity, partitioning, schema, nulls | Raw risks documented |
| Silver | Cleaning, standardization, type consistency | Cleaned tables validated |
| Gold | Feature engineering and target readiness | Model-ready dataset |

## Progress

Check each layer's `checkpoints/` folder for completion markers.

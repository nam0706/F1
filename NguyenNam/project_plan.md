 # F1 WinRate Predictor - ETL Pipeline Implementation Plan

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mục tiêu
Xây dựng pipeline ETL cho dữ liệu F1 để dự đoán **win probability** và **vị trí top 3/5/10** cho mỗi tay đua trong mỗi race.

### 1.2 Dữ liệu đầu vào
- Thư mục: `data/raw/`
- 13 files: meetings.csv, sessions.csv, drivers.csv, session_results.csv, laps.csv, weather.csv, stints.csv, starting_grid.csv, intervals.csv, position.csv, overtakes.csv, car_data.parquet, location.parquet

### 1.3 Dữ liệu đầu ra

**Silver layer** (`data/cleaned/`):
- 13 files ở định dạng Parquet (đã làm sạch)
- Chú ý: car_data và location vẫn giữ nguyên cấu trúc nhưng đã được validate và thêm cột lap_number

**Gold layer** (`data/features/`):
- 6 files Parquet riêng biệt:
  - `driver_race_features.parquet`
  - `driver_telemetry_features.parquet`
  - `team_features.parquet`
  - `race_context_features.parquet`
  - `qualifying_features.parquet`
  - `target_labels.parquet`

### 1.4 Công nghệ sử dụng
- **Python 3.9+**
- **Pandas** (cho small files)
- **PyArrow** (cho Parquet và streaming)
- **DuckDB** (cho query trên Parquet files)
- **Matplotlib/Seaborn** (cho visualization)
- **Jupyter Notebook** (cho EDA)

### 1.5 Cài đặt dependencies
```bash
pip install pandas numpy pyarrow duckdb matplotlib seaborn jupyter scikit-learn tqdm
2. CẤU TRÚC THƯ MỤC
2.1 Script tạo thư mục
Tạo file scripts/create_structure.py:

python
import os
from pathlib import Path

def create_directory_structure():
    """Create complete project directory structure"""
    
    directories = [
        # Data directories
        "data/raw",
        "data/cleaned",
        "data/features",
        
        # EDA shared
        "eda/shared/scripts",
        "eda/shared/logs",
        
        # Bronze layer
        "eda/bronze/notebooks",
        "eda/bronze/outputs/tables",
        "eda/bronze/outputs/charts",
        "eda/bronze/outputs/reports",
        "eda/bronze/insights",
        "eda/bronze/checkpoints",
        
        # Silver layer
        "eda/silver/notebooks",
        "eda/silver/outputs/tables",
        "eda/silver/outputs/charts",
        "eda/silver/outputs/reports",
        "eda/silver/insights",
        "eda/silver/checkpoints",
        
        # Gold layer
        "eda/gold/notebooks",
        "eda/gold/outputs/tables",
        "eda/gold/outputs/charts",
        "eda/gold/outputs/reports",
        "eda/gold/insights",
        "eda/gold/checkpoints",
        
        # Other
        "scripts",
        "reports/validation",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n" + "="*50)
    print("✅ Directory structure created successfully!")
    print("="*50)

if __name__ == "__main__":
    create_directory_structure()
2.2 File tree cuối cùng
text
F1_WinRate_Predictor/
├── data/
│   ├── raw/                      # Dữ liệu crawl (đã có)
│   ├── cleaned/                  # Silver layer (sẽ tạo)
│   └── features/                 # Gold layer (sẽ tạo)
│
├── eda/
│   ├── shared/
│   │   ├── scripts/
│   │   │   ├── config.py
│   │   │   ├── file_utils.py
│   │   │   ├── validation_utils.py
│   │   │   ├── cleaning_utils.py
│   │   │   ├── feature_utils.py
│   │   │   └── viz_utils.py
│   │   └── logs/
│   │       └── pipeline.log
│   │
│   ├── bronze/                   # VALIDATION PHASE
│   │   ├── notebooks/
│   │   │   ├── 01_file_integrity.ipynb
│   │   │   ├── 02_schema_validation.ipynb
│   │   │   ├── 03_pk_fk_checks.ipynb
│   │   │   ├── 04_null_analysis.ipynb
│   │   │   ├── 05_range_validation.ipynb
│   │   │   ├── 06_temporal_checks.ipynb
│   │   │   └── 00_final_summary.ipynb
│   │   ├── outputs/
│   │   │   ├── tables/
│   │   │   ├── charts/
│   │   │   └── reports/
│   │   ├── insights/
│   │   │   ├── 01_file_integrity_insights.md
│   │   │   ├── 02_schema_insights.md
│   │   │   ├── 03_pk_fk_insights.md
│   │   │   ├── 04_null_insights.md
│   │   │   ├── 05_range_insights.md
│   │   │   ├── 06_temporal_insights.md
│   │   │   └── 00_final_summary.md
│   │   └── checkpoints/
│   │       └── bronze_completed.txt
│   │
│   ├── silver/                   # CLEANING PHASE
│   │   ├── notebooks/
│   │   │   ├── 01_handle_nulls.ipynb
│   │   │   ├── 02_remove_duplicates.ipynb
│   │   │   ├── 03_fix_data_types.ipynb
│   │   │   ├── 04_standardize_text.ipynb
│   │   │   ├── 05_handle_outliers.ipynb
│   │   │   ├── 06_add_lap_number_telemetry.ipynb
│   │   │   └── 07_save_cleaned_data.ipynb
│   │   ├── outputs/
│   │   │   ├── tables/
│   │   │   ├── charts/
│   │   │   └── reports/
│   │   ├── insights/
│   │   │   ├── 01_cleaning_summary.md
│   │   │   └── 00_final_quality_report.md
│   │   └── checkpoints/
│   │       └── silver_completed.txt
│   │
│   └── gold/                     # FEATURE ENGINEERING
│       ├── notebooks/
│       │   ├── 01_driver_race_features.ipynb
│       │   ├── 02_driver_telemetry_features.ipynb
│       │   ├── 03_team_features.ipynb
│       │   ├── 04_race_context_features.ipynb
│       │   ├── 05_qualifying_features.ipynb
│       │   ├── 06_target_labels.ipynb
│       │   └── 07_merge_and_export.ipynb
│       ├── outputs/
│       │   ├── tables/
│       │   ├── charts/
│       │   └── reports/
│       ├── insights/
│       │   ├── 01_feature_analysis.md
│       │   └── 00_feature_specification.md
│       └── checkpoints/
│           └── gold_completed.txt
│
├── scripts/
│   ├── create_structure.py
│   ├── run_bronze.py
│   ├── run_silver.py
│   ├── run_gold.py
│   └── run_pipeline.py
│
├── reports/
│   └── validation/
│
└── models/
    └── (sẽ chứa model sau khi train)
3. PHASE 1: BRONZE LAYER - DATA VALIDATION
3.1 Priority Levels
Priority	Màu	Ý nghĩa	Hành động
P0	🔴 Đỏ	Critical - Dữ liệu không dùng được	PHẢI PASS, nếu FAIL thì dừng pipeline
P1	🟡 Vàng	High - Ảnh hưởng feature engineering	WARNING, có thể tiếp tục nhưng cần ghi nhận
P2	🔵 Xanh	Medium - Ảnh hưởng chất lượng model	INFO, chỉ để tham khảo
P3	⚪ Trắng	Low - Optimization	OPTIONAL, không ảnh hưởng
3.2 File cấu hình chung
File: eda/shared/scripts/config.py

python
from pathlib import Path

# ============================================
# PATH CONFIGURATION
# ============================================

RAW_DATA_PATH = Path("data/raw")
CLEANED_DATA_PATH = Path("data/cleaned")
FEATURES_DATA_PATH = Path("data/features")

# ============================================
# FILE LISTS
# ============================================

EXPECTED_FILES = [
    'meetings.csv',
    'sessions.csv',
    'drivers.csv',
    'session_results.csv',
    'laps.csv',
    'weather.csv',
    'stints.csv',
    'starting_grid.csv',
    'intervals.csv',
    'position.csv',
    'overtakes.csv',
    'car_data.parquet',
    'location.parquet'
]

TELEMETRY_FILES = ['car_data.parquet', 'location.parquet']

# ============================================
# CRITICAL COLUMNS (must have 0% nulls)
# ============================================

CRITICAL_COLUMNS = {
    'meetings.csv': ['meeting_key', 'meeting_name', 'date_start', 'date_end'],
    'sessions.csv': ['session_key', 'meeting_key', 'date_start', 'date_end'],
    'drivers.csv': ['driver_number', 'full_name', 'team_name'],
    'laps.csv': ['session_key', 'driver_number', 'lap_number', 'lap_duration'],
    'session_results.csv': ['session_key', 'driver_number'],
    'car_data.parquet': ['session_key', 'driver_number', 'date_time'],
    'location.parquet': ['session_key', 'driver_number', 'date_time'],
}

# ============================================
# PRIMARY KEYS
# ============================================

PRIMARY_KEYS = {
    'meetings.csv': ['meeting_key'],
    'sessions.csv': ['session_key'],
    'drivers.csv': ['meeting_key', 'driver_number'],
    'laps.csv': ['session_key', 'driver_number', 'lap_number'],
    'session_results.csv': ['session_key', 'driver_number'],
    'starting_grid.csv': ['session_key', 'driver_number'],
    'stints.csv': ['session_key', 'driver_number', 'stint_number'],
    'weather.csv': ['session_key', 'date_time'],
    'overtakes.csv': ['session_key', 'date_time', 'driver_number', 'overtaken_driver'],
    'position.csv': ['session_key', 'driver_number', 'date_time'],
    'intervals.csv': ['session_key', 'date_time']
}

# ============================================
# FOREIGN KEYS
# ============================================

FOREIGN_KEYS = [
    {'child': 'sessions.csv', 'child_key': 'meeting_key', 'parent': 'meetings.csv', 'parent_key': 'meeting_key'},
    {'child': 'laps.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'session_results.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'starting_grid.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'weather.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'stints.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'intervals.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'position.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'overtakes.csv', 'child_key': 'session_key', 'parent': 'sessions.csv', 'parent_key': 'session_key'},
    {'child': 'laps.csv', 'child_key': 'driver_number', 'parent': 'drivers.csv', 'parent_key': 'driver_number'},
    {'child': 'session_results.csv', 'child_key': 'driver_number', 'parent': 'drivers.csv', 'parent_key': 'driver_number'},
]

# ============================================
# EXPECTED SCHEMAS
# ============================================

EXPECTED_SCHEMAS = {
    'meetings.csv': {
        'columns': ['meeting_key', 'meeting_name', 'country', 'circuit_short_name', 
                    'date_start', 'date_end', 'gmt_offset', 'year'],
        'dtypes': {
            'meeting_key': 'int64',
            'meeting_name': 'object',
            'country': 'object',
            'circuit_short_name': 'object',
            'date_start': 'datetime64[ns]',
            'date_end': 'datetime64[ns]',
            'gmt_offset': 'float64',
            'year': 'int64'
        }
    },
    'sessions.csv': {
        'columns': ['session_key', 'meeting_key', 'session_name', 'session_type', 
                    'date_start', 'date_end', 'year'],
        'dtypes': {
            'session_key': 'int64',
            'meeting_key': 'int64',
            'session_name': 'object',
            'session_type': 'object',
            'date_start': 'datetime64[ns]',
            'date_end': 'datetime64[ns]',
            'year': 'int64'
        }
    },
    'drivers.csv': {
        'columns': ['driver_number', 'full_name', 'team_name', 'country_code', 'meeting_key'],
        'dtypes': {
            'driver_number': 'int64',
            'full_name': 'object',
            'team_name': 'object',
            'country_code': 'object',
            'meeting_key': 'int64'
        }
    },
    'laps.csv': {
        'columns': ['session_key', 'driver_number', 'lap_number', 'lap_duration', 
                    'sector1_time', 'sector2_time', 'sector3_time', 
                    'pit_out_time', 'pit_in_time', 'deleted'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'lap_number': 'int64',
            'lap_duration': 'float64',
            'sector1_time': 'float64',
            'sector2_time': 'float64',
            'sector3_time': 'float64',
            'pit_out_time': 'float64',
            'pit_in_time': 'float64',
            'deleted': 'bool'
        }
    },
    'session_results.csv': {
        'columns': ['session_key', 'driver_number', 'position', 'grid_position', 
                    'time', 'status', 'points', 'laps'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'position': 'object',
            'grid_position': 'int64',
            'time': 'object',
            'status': 'object',
            'points': 'float64',
            'laps': 'int64'
        }
    },
    'weather.csv': {
        'columns': ['session_key', 'date_time', 'air_temperature', 'track_temperature', 
                    'humidity', 'pressure', 'wind_speed', 'rainfall'],
        'dtypes': {
            'session_key': 'int64',
            'date_time': 'datetime64[ns]',
            'air_temperature': 'float64',
            'track_temperature': 'float64',
            'humidity': 'float64',
            'pressure': 'float64',
            'wind_speed': 'float64',
            'rainfall': 'bool'
        }
    },
    'stints.csv': {
        'columns': ['session_key', 'driver_number', 'stint_number', 'lap_start', 
                    'lap_end', 'compound', 'new'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'stint_number': 'int64',
            'lap_start': 'int64',
            'lap_end': 'int64',
            'compound': 'object',
            'new': 'bool'
        }
    },
    'starting_grid.csv': {
        'columns': ['session_key', 'driver_number', 'position'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'position': 'int64'
        }
    },
    'intervals.csv': {
        'columns': ['session_key', 'date_time', 'gap_to_leader', 'gap_to_car_ahead', 'interval_type'],
        'dtypes': {
            'session_key': 'int64',
            'date_time': 'datetime64[ns]',
            'gap_to_leader': 'float64',
            'gap_to_car_ahead': 'float64',
            'interval_type': 'object'
        }
    },
    'position.csv': {
        'columns': ['session_key', 'driver_number', 'date_time', 'position', 'status'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'date_time': 'datetime64[ns]',
            'position': 'int64',
            'status': 'object'
        }
    },
    'overtakes.csv': {
        'columns': ['session_key', 'driver_number', 'overtaken_driver', 'lap_number', 
                    'date_time', 'position'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'overtaken_driver': 'int64',
            'lap_number': 'int64',
            'date_time': 'datetime64[ns]',
            'position': 'int64'
        }
    },
    'car_data.parquet': {
        'columns': ['session_key', 'driver_number', 'date_time', 'speed', 'rpm', 
                    'gear', 'throttle', 'brake', 'drs'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'date_time': 'datetime64[ns]',
            'speed': 'float64',
            'rpm': 'int64',
            'gear': 'int64',
            'throttle': 'float64',
            'brake': 'float64',
            'drs': 'int64'
        }
    },
    'location.parquet': {
        'columns': ['session_key', 'driver_number', 'date_time', 'x', 'y', 'z'],
        'dtypes': {
            'session_key': 'int64',
            'driver_number': 'int64',
            'date_time': 'datetime64[ns]',
            'x': 'float64',
            'y': 'float64',
            'z': 'float64'
        }
    }
}

# ============================================
# RANGE VALIDATION RULES
# ============================================

RANGE_RULES = {
    'driver_number': {'min': 1, 'max': 99, 'allow_null': False},
    'position': {'min': 0, 'max': 20, 'allow_null': False},
    'grid_position': {'min': 1, 'max': 20, 'allow_null': False},
    'points': {'min': 0, 'max': 26, 'allow_null': True},
    'laps': {'min': 1, 'allow_null': False},
    'lap_duration': {'min': 60, 'max': 600, 'allow_null': False},
    'speed': {'min': 0, 'max': 380, 'allow_null': True},
    'rpm': {'min': 0, 'max': 15000, 'allow_null': True},
    'gear': {'min': -1, 'max': 8, 'allow_null': True},
    'throttle': {'min': 0, 'max': 100, 'allow_null': True},
    'brake': {'min': 0, 'max': 100, 'allow_null': True},
    'drs': {'min': 0, 'max': 2, 'allow_null': True},
    'air_temperature': {'min': -10, 'max': 50, 'allow_null': True},
    'track_temperature': {'min': -10, 'max': 70, 'allow_null': True},
    'humidity': {'min': 0, 'max': 100, 'allow_null': True},
    'pressure': {'min': 900, 'max': 1050, 'allow_null': True},
    'wind_speed': {'min': 0, 'allow_null': True},
    'gap_to_leader': {'min': 0, 'allow_null': True},
    'gap_to_car_ahead': {'min': 0, 'allow_null': True},
    'gmt_offset': {'min': -12, 'max': 14, 'allow_null': True},
    'year': {'min': 1950, 'max': 2025, 'allow_null': False}
}

# ============================================
# VALID VALUES FOR CATEGORICAL COLUMNS
# ============================================

VALID_VALUES = {
    'session_name': ['FP1', 'FP2', 'FP3', 'Q', 'Q1', 'Q2', 'Q3', 'Race', 'Sprint', 'Sprint Shootout'],
    'session_type': ['Practice', 'Qualifying', 'Race', 'Sprint', 'Sprint Qualifying'],
    'compound': ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'UNKNOWN'],
    'interval_type': ['LAP_TIME', 'GAP', 'PIT', 'SC', 'VSC'],
    'status': ['ON_TRACK', 'PIT', 'OFF_TRACK', 'OUT_OF_SESSION']
}

# ============================================
# THRESHOLDS
# ============================================

THRESHOLDS = {
    'max_null_percentage_critical': 0,
    'max_null_percentage_warning': 30,
    'max_duplicate_percentage': 0.1,
    'min_rows_per_session': 100,
    'max_telemetry_time_gap_seconds': 5,
    'max_speed_change_per_100ms': 50,
    'max_position_change_per_lap': 19
}

# ============================================
# TELEMETRY STREAMING CONFIG
# ============================================

TELEMETRY_CONFIG = {
    'batch_size_rows': 500000,
    'max_time_gap_seconds': 5,
    'expected_sampling_rate_hz': 20,
    'sampling_rate_tolerance': 0.05
}
4. CHI TIẾT CÁC NOTEBOOKS BRONZE
Notebook 1: 01_file_integrity.ipynb
Mục đích: Kiểm tra file tồn tại, kích thước, không bị corrupt

P0 Checks:

File có tồn tại trong data/raw/ không? (13/13 files)

File có kích thước > 0 bytes không?

File có thể mở được không (không bị corrupt)?

Output files:

outputs/tables/file_integrity.csv - Bảng tổng hợp

outputs/charts/file_sizes.png - Biểu đồ kích thước

outputs/reports/file_integrity.json - JSON report

insights/01_file_integrity_insights.md - Nhận xét

Notebook 2: 02_schema_validation.ipynb
Mục đích: Kiểm tra cấu trúc cột, kiểu dữ liệu

P0 Checks:

Số lượng columns có khớp với expected schema không?

Các cột bắt buộc (critical columns) có tồn tại không?

Primary key columns có tồn tại không?

P1 Checks:

Data types có đúng không? (datetime, int, float, string)

Output files:

outputs/tables/schema_validation.csv

outputs/charts/missing_columns_heatmap.png

outputs/reports/schema_validation.json

insights/02_schema_insights.md

Notebook 3: 03_pk_fk_checks.ipynb
Mục đích: Kiểm tra ràng buộc khóa chính và khóa ngoại

P0 Checks:

Primary key có unique không? (không duplicate)

Foreign key references có tồn tại trong parent table không? (không orphan records)

Output files:

outputs/tables/pk_fk_validation.csv

outputs/charts/duplicate_counts.png

outputs/reports/pk_fk_validation.json

insights/03_pk_fk_insights.md

Notebook 4: 04_null_analysis.ipynb
Mục đích: Phân tích giá trị null/empty

P0 Checks:

Critical columns có null không? (must be 0% null)

Có cột nào 100% null không? (cần drop)

P1 Checks:

Tỷ lệ null của từng cột (< 30% là OK)

Output files:

outputs/tables/null_percentage.csv

outputs/charts/null_heatmap.png

outputs/reports/null_analysis.json

insights/04_null_insights.md

Notebook 5: 05_range_validation.ipynb
Mục đích: Kiểm tra giá trị có nằm trong range hợp lý không

Output files:

outputs/tables/range_violations.csv

outputs/charts/outliers_boxplot.png

outputs/reports/range_validation.json

insights/05_range_insights.md

Notebook 6: 06_temporal_checks.ipynb
Mục đích: Kiểm tra tính hợp lý về thời gian

Output files:

outputs/tables/temporal_issues.csv

outputs/charts/sampling_rate_distribution.png

outputs/reports/temporal_validation.json

insights/06_temporal_insights.md

Notebook 7: 00_final_summary.ipynb
Mục đích: Gộp tất cả kết quả validation, quyết định PASS/FAIL

Output files:

outputs/reports/bronze_validation_complete.json

insights/00_final_summary.md

checkpoints/bronze_completed.txt (chỉ tạo nếu tất cả P0 checks PASS)

5. STREAMING PATTERN CHO TELEMETRY FILES
Quan trọng: Không được đọc toàn bộ car_data.parquet (65M rows) và location.parquet (33M rows) vào RAM!

Code pattern bắt buộc:

python
import pyarrow.parquet as pq

def process_telemetry_streaming(file_path, columns_needed, processing_function):
    """
    Process large parquet files using streaming row groups
    
    Args:
        file_path: Path to parquet file
        columns_needed: List of columns to read
        processing_function: Function to apply to each batch
    """
    pf = pq.ParquetFile(file_path)
    
    results = []
    
    for rg_idx in range(pf.num_row_groups):
        # Only read current row group, not entire file
        table = pf.read_row_group(rg_idx, columns=columns_needed)
        df = table.to_pandas()
        
        # Process this batch
        batch_result = processing_function(df)
        results.append(batch_result)
        
        # Clear memory
        del df, table
    
    return results
6. CHI TIẾT CÁC NOTEBOOKS SILVER
Notebook 1: 01_handle_nulls.ipynb
Xử lý null values theo chiến lược:

Table	Column	Strategy	Reason
laps.csv	sector1_time, sector2_time, sector3_time	Fill with median	Timing data
laps.csv	pit_in_time, pit_out_time	Fill with 0	No pit stop
weather.csv	all columns	Forward fill	Weather continuous
car_data.parquet	throttle, brake	Fill with 0	Sensor default
car_data.parquet	drs	Fill with 0	DRS not available
position.csv	status	Fill with 'ON_TRACK'	Default status
Notebook 2: 02_remove_duplicates.ipynb
python
# Drop duplicates based on primary key
df = df.drop_duplicates(subset=PRIMARY_KEYS[table_name], keep='first')
Notebook 3: 03_fix_data_types.ipynb
python
dtype_conversions = {
    'session_key': 'int32',
    'driver_number': 'int16',
    'date_time': 'datetime64[ns]',
    'date_start': 'datetime64[ns]',
    'date_end': 'datetime64[ns]',
    'lap_duration': 'float32',
    'speed': 'float32',
    'rpm': 'int32',
    'gear': 'int8',
    'throttle': 'float32',
    'brake': 'float32',
}
Notebook 4: 04_standardize_text.ipynb
python
team_mapping = {
    'Red Bull Racing': 'Red Bull',
    'Red Bull': 'Red Bull',
    'Oracle Red Bull Racing': 'Red Bull',
    'Mercedes-AMG Petronas': 'Mercedes',
    'Mercedes': 'Mercedes',
    'Ferrari': 'Ferrari',
    'Scuderia Ferrari': 'Ferrari',
}
Notebook 5: 05_handle_outliers.ipynb
python
# Clip outliers to valid ranges
df['speed'] = df['speed'].clip(lower=0, upper=380)
df['rpm'] = df['rpm'].clip(lower=0, upper=15000)
df['throttle'] = df['throttle'].clip(lower=0, upper=100)
df['brake'] = df['brake'].clip(lower=0, upper=100)
Notebook 6: 06_add_lap_number_telemetry.ipynb ⭐ QUAN TRỌNG
Thêm cột lap_number vào car_data và location

Logic:

Load laps.csv để biết timing của từng lap

Với mỗi session_key, mỗi driver_number:

Xác định timestamp bắt đầu và kết thúc mỗi lap

Gán lap_number cho mỗi telemetry row dựa trên timestamp

python
def add_lap_number_to_telemetry_streaming(telemetry_path, laps_df):
    """
    Add lap_number to telemetry data using streaming
    
    Args:
        telemetry_path: Path to car_data.parquet or location.parquet
        laps_df: DataFrame with lap timing information
    
    Returns:
        Path to cleaned file with lap_number added
    """
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    pf = pq.ParquetFile(telemetry_path)
    output_path = telemetry_path.parent / f"{telemetry_path.stem}_with_laps.parquet"
    
    writer = None
    
    for rg_idx in range(pf.num_row_groups):
        df = pf.read_row_group(rg_idx).to_pandas()
        
        # Add lap_number column
        df['lap_number'] = 0
        
        for (session_key, driver_number), group in df.groupby(['session_key', 'driver_number']):
            driver_laps = laps_df[
                (laps_df['session_key'] == session_key) & 
                (laps_df['driver_number'] == driver_number)
            ]
            
            for _, lap in driver_laps.iterrows():
                mask = (group['date_time'] >= lap['date_start']) & \
                       (group['date_time'] <= lap['date_end'])
                df.loc[group[mask].index, 'lap_number'] = lap['lap_number']
        
        # Write to new parquet
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        
        del df, table
    
    if writer:
        writer.close()
    
    return output_path
Notebook 7: 07_save_cleaned_data.ipynb
Lưu toàn bộ cleaned data

Đảm bảo tất cả files đã được xử lý

Lưu metadata về quá trình cleaning

Tạo checkpoint silver_completed.txt

7. CHI TIẾT CÁC NOTEBOOKS GOLD
File 1: data/features/driver_race_features.parquet
Features:

Feature	Description	Source
avg_lap_time_seconds	Average lap time	laps.csv
std_lap_time_seconds	Lap time consistency	laps.csv
fastest_lap_seconds	Fastest lap time	laps.csv
laps_completed	Total laps completed	laps.csv
overtakes_made	Number of overtakes	overtakes.csv
net_positions_gained	(start_position - finish_position)	starting_grid + session_results
pit_stops_count	Number of pit stops	laps.csv
qualifying_position	Grid position	starting_grid.csv
File 2: data/features/driver_telemetry_features.parquet
Features (từ car_data):

Feature	Description
avg_speed_kmh	Average speed over all laps
max_speed_kmh	Top speed
avg_throttle_pct	Average throttle application
avg_brake_pct	Average brake application
avg_rpm	Average engine RPM
drs_usage_count	Number of times DRS opened
File 3: data/features/team_features.parquet
Features:

Feature	Description
team_avg_pit_stop_time	Average pit stop duration
team_best_pit_stop	Fastest pit stop
team_dnf_count	Number of DNFs in team
team_avg_finish_position	Average finish position
team_points_total	Total points scored
File 4: data/features/race_context_features.parquet
Features:

Feature	Source
avg_air_temperature	weather.csv
avg_track_temperature	weather.csv
rainfall_during_race	weather.csv
total_safety_car_laps	position.csv
File 5: data/features/qualifying_features.parquet
Features:

Feature	Description
q1_time	Best time in Q1
q2_time	Best time in Q2
q3_time	Best time in Q3
qualifying_position	Final qualifying position
File 6: data/features/target_labels.parquet
Target variables:

Column	Description	Values
win_binary	Won the race?	0/1
position_category	Finish position category	1=Top3, 2=Top5, 3=Top10, 4=Bottom
points_scored	Points earned	0-26
8. QUALITY GATES & CHECKPOINTS
Gate 1: Bronze Completion
Điều kiện: Tất cả P0 checks PASS
Tạo file: eda/bronze/checkpoints/bronze_completed.txt

python
# Content of bronze_completed.txt
{
    "status": "PASSED",
    "timestamp": "2026-01-15T14:30:22",
    "p0_checks_passed": 5,
    "p0_checks_total": 5,
    "p1_warnings": 2,
    "next_step": "Proceed to Silver layer"
}
Gate 2: Silver Completion
Điều kiện: Tất cả cleaning steps completed
Tạo file: eda/silver/checkpoints/silver_completed.txt

Gate 3: Gold Completion
Điều kiện: Tất cả feature files created
Tạo file: eda/gold/checkpoints/gold_completed.txt

9. SCRIPT CHẠY TOÀN BỘ PIPELINE
File: scripts/run_pipeline.py

python
#!/usr/bin/env python3
"""
Complete ETL Pipeline Runner for F1 WinRate Predictor
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_notebook(notebook_path):
    """Run a Jupyter notebook and convert to HTML"""
    result = subprocess.run([
        "jupyter", "nbconvert", "--to", "html",
        "--execute", str(notebook_path),
        "--output-dir", str(notebook_path.parent.parent / "outputs" / "reports")
    ], capture_output=True, text=True)
    
    return result.returncode == 0

def check_checkpoint(checkpoint_path):
    """Check if a checkpoint file exists"""
    return Path(checkpoint_path).exists()

def main():
    print("="*60)
    print("F1 WinRate Predictor - ETL Pipeline")
    print(f"Start time: {datetime.now()}")
    print("="*60)
    
    # Phase 1: Bronze Validation
    print("\n[Phase 1] Running Bronze Validation...")
    
    bronze_notebooks = sorted(Path("eda/bronze/notebooks").glob("*.ipynb"))
    
    for nb in bronze_notebooks:
        print(f"  Running {nb.name}...")
        if not run_notebook(nb):
            print(f"  ❌ Failed: {nb.name}")
            sys.exit(1)
        print(f"  ✅ Completed: {nb.name}")
    
    # Check if bronze passed
    if not check_checkpoint("eda/bronze/checkpoints/bronze_completed.txt"):
        print("\n❌ Bronze validation failed. Pipeline stopped.")
        sys.exit(1)
    
    print("\n✅ Bronze validation PASSED!")
    
    # Phase 2: Silver Cleaning
    print("\n[Phase 2] Running Silver Cleaning...")
    
    silver_notebooks = sorted(Path("eda/silver/notebooks").glob("*.ipynb"))
    
    for nb in silver_notebooks:
        print(f"  Running {nb.name}...")
        if not run_notebook(nb):
            print(f"  ❌ Failed: {nb.name}")
            sys.exit(1)
        print(f"  ✅ Completed: {nb.name}")
    
    if not check_checkpoint("eda/silver/checkpoints/silver_completed.txt"):
        print("\n❌ Silver cleaning failed. Pipeline stopped.")
        sys.exit(1)
    
    print("\n✅ Silver cleaning PASSED!")
    
    # Phase 3: Gold Feature Engineering
    print("\n[Phase 3] Running Gold Feature Engineering...")
    
    gold_notebooks = sorted(Path("eda/gold/notebooks").glob("*.ipynb"))
    
    for nb in gold_notebooks:
        print(f"  Running {nb.name}...")
        if not run_notebook(nb):
            print(f"  ❌ Failed: {nb.name}")
            sys.exit(1)
        print(f"  ✅ Completed: {nb.name}")
    
    if not check_checkpoint("eda/gold/checkpoints/gold_completed.txt"):
        print("\n❌ Gold feature engineering failed.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"End time: {datetime.now()}")
    print("="*60)
    
    print("\nOutput files:")
    print("  - Cleaned data: data/cleaned/")
    print("  - Features: data/features/")
    print("  - Reports: eda/*/outputs/reports/")
    print("  - Insights: eda/*/insights/")

if __name__ == "__main__":
    main()
10. EXECUTION COMMANDS
bash
# Step 1: Create directory structure
python scripts/create_structure.py

# Step 2: Run entire pipeline
python scripts/run_pipeline.py

# Or run each phase manually:
# Phase 1: Bronze
cd eda/bronze/notebooks
jupyter notebook 01_file_integrity.ipynb
# ... run all notebooks in order

# Phase 2: Silver
cd eda/silver/notebooks
jupyter notebook 01_handle_nulls.ipynb
# ... run all notebooks in order

# Phase 3: Gold
cd eda/gold/notebooks
jupyter notebook 01_driver_race_features.ipynb
# ... run all notebooks in order
11. SUCCESS CRITERIA
Phase	Success Criteria	Output
Bronze	All P0 checks PASS	bronze_completed.txt
Silver	All data cleaned and saved	13 Parquet files in data/cleaned/
Gold	All feature files created	6 Parquet files in data/features/
END OF PLAN
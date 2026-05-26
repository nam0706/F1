# Data Dictionary

## Bảng dữ liệu chính

| File | Grain | Khóa chính | Mục đích |
|---|---|---|---|
| `data/processed/driver_session_base.csv` | Driver-session | `session_key + driver_number` | Bộ dữ liệu chính dùng cho feature engineering và huấn luyện mô hình machine learning. |

---

## Báo cáo chất lượng dữ liệu

| File | Mục đích |
|---|---|
| `reports/data_quality/cleaning_summary.md` | Tóm tắt số lượng dòng dữ liệu được đọc, giữ lại và loại bỏ trong từng bước cleaning. |
| `reports/data_quality/validation_report.md` | Kết quả validation về schema, khóa chính, missing values và kiểm tra phạm vi dữ liệu. |
| `reports/data_quality/base_dataset_report.md` | Báo cáo tổng quan về kích thước và cấu trúc của bộ dữ liệu sau xử lý. |

---

# Codebook

## Các cột dữ liệu chính

| Column | Description | Data Type | Format | Value Range | Unit |
|---|---|---|---|---|---|
| `session_key` | Mã định danh duy nhất của mỗi phiên đua từ OpenF1 API. | Integer | Numeric ID | > 0 | None |
| `driver_number` | Số xe chính thức của tay đua Formula 1. | String | Character string | Ví dụ: `"1"`, `"44"` | None |
| `year` | Năm diễn ra mùa giải Formula 1. | Integer | `YYYY` | Ví dụ: `2024` | Year |
| `session_type` | Loại phiên đua trong tuần thi đấu. | String | Text | `Race`, `Qualifying`, `Sprint`, ... | None |
| `circuit_key` | Mã định danh duy nhất của đường đua. | Integer | Numeric ID | > 0 | None |
| `grid_position` | Vị trí xuất phát của tay đua trong phiên đua. | Integer | Integer | 1–20 | Position |
| `final_position` | Vị trí cuối cùng của tay đua khi kết thúc phiên đua. | Integer | Integer | 1–20 | Position |
| `target_win` | Binary target label cho biết tay đua có chiến thắng cuộc đua hay không. | Integer (Binary) | `0` hoặc `1` | [0,1] | Classification Label |
| `target_podium` | Binary target label cho biết tay đua có kết thúc trong top 3 hay không. | Integer (Binary) | `0` hoặc `1` | [0,1] | Classification Label |
| `target_top10` | Binary target label cho biết tay đua có kết thúc trong top 10 hay không. | Integer (Binary) | `0` hoặc `1` | [0,1] | Classification Label |
| `avg_lap_duration` | Thời gian hoàn thành trung bình mỗi vòng đua của tay đua trong phiên đua. | Float | Decimal number | > 0 | Seconds |
| `best_lap_duration` | Thời gian hoàn thành vòng đua nhanh nhất của tay đua trong phiên đua. | Float | Decimal number | > 0 | Seconds |
| `air_temperature_mean` | Nhiệt độ không khí trung bình ghi nhận trong phiên đua. | Float | Decimal number | Phụ thuộc điều kiện thời tiết | °C |
| `track_temperature_mean` | Nhiệt độ bề mặt đường đua trung bình ghi nhận trong phiên đua. | Float | Decimal number | Phụ thuộc điều kiện thời tiết | °C |
| `rainfall_max` | Chỉ số lượng mưa lớn nhất ghi nhận trong phiên đua. | Float / Binary | Decimal hoặc binary value | Thường từ 0–1 | Weather Indicator |

---

## Ghi chú

- Grain của dataset được xác định ở mức **driver-session**, nghĩa là mỗi dòng dữ liệu đại diện cho một tay đua trong một phiên đua.
- Các biến `target_win`, `target_podium` và `target_top10` được sử dụng làm target variables cho bài toán supervised machine learning classification.
- Các đặc trưng liên quan đến lap duration được tính theo đơn vị giây và được tổng hợp ở mức driver-session.
- Các biến thời tiết được tổng hợp từ telemetry và environmental data của phiên đua.
"""
CONFIGURATION & HYPERPARAMETERS — ISOLATION FOREST PROJECT
====================================================
File tập trung toàn bộ hằng số và cấu hình siêu tham số.
Tách ra để dễ điều chỉnh mà không cần đụng vào logic chính.
"""

# ==============================================================================
# 1. CẤU HÌNH CHẠY MẶC ĐỊNH (SINGLE-RUN PIPELINE)
# ==============================================================================

DEFAULT_CONFIG = {
    # --- Dữ liệu ---
    "data_path":     "shuttle.csv",     # Đường dẫn dataset
    "test_size":     0.2,               # 20% tập kiểm thử

    # --- Mô hình ---
    "n_estimators":  100,               # Số cây iTree trong rừng
    "max_samples":   256,               # Kích thước mẫu con psi (Liu et al., 2008 mặc định 256)
    "max_features":  1.0,               # Tỷ lệ đặc trưng Feature Bagging (1.0 = dùng tất cả)
    "contamination": "auto",            # "auto" → ngưỡng lý thuyết 0.5; float ∈ (0,0.5) → phân vị

    # --- Tái lập ---
    "random_state":  42,                # Seed ngẫu nhiên
}


# ==============================================================================
# 2. LƯỚI SIÊU THAM SỐ — UNSUPERVISED K-FOLD CV TUNING  (--tune)
# ==============================================================================
# Tiêu chí lựa chọn: Cấu hình có Score Spread (Std anomaly score) LỚN NHẤT
# trên tập validation → phân biệt tốt nhất normal vs anomaly mà không cần nhãn.

PARAM_GRID = [
    {"n_estimators": 100, "max_samples": 128, "max_features": 1.0},   # Baseline nhỏ
    {"n_estimators": 100, "max_samples": 256, "max_features": 1.0},   # Mặc định Liu et al. (2008)
    {"n_estimators": 150, "max_samples": 256, "max_features": 0.8},   # Feature Bagging 80%
    {"n_estimators": 200, "max_samples": 256, "max_features": 1.0},   # Nhiều cây hơn
]

# Số fold trong cross-validation không giám sát
KFOLD_N_SPLITS: int = 3


# ==============================================================================
# 3. CẤU HÌNH NGƯỠNG PHÁT HIỆN BẤT THƯỜNG (UNSUPERVISED THRESHOLDS)
# ==============================================================================

THRESHOLD_CONFIG = {
    # Ngưỡng lý thuyết từ bài báo gốc Liu et al. (2008): s >= 0.5 → anomaly
    "theoretical":   0.5,

    # Các phân vị trên tập TRAIN (ước lượng contamination không giám sát)
    "top5_pct":      95.0,   # Phân vị 95 → ~5%  mẫu được coi là bất thường
    "top1_pct":      99.0,   # Phân vị 99 → ~1%  mẫu cực kỳ bất thường
    "top01_pct":     99.9,   # Phân vị 99.9 → 0.1% cực đoan

    # Ngưỡng thống kê: Mean + k*Std (Gaussian approximation)
    "stat_sigma_k":   2.0,   # k = 2 → ~2.275% tail dưới phân phối chuẩn
}

# Nhãn mức cảnh báo hiển thị trong báo cáo RCA
ALERT_LEVELS = {
    "high_danger":  0.60,    # score >= 0.60 → "NGUY HIỂM CAO"
    "warning":      0.50,    # score >= 0.50 → "CẢNH BÁO BẤT THƯỜNG"
}


# ==============================================================================
# 4. TRỌNG SỐ TỔNG HỢP — FEATURE IMPORTANCE
# ==============================================================================
# Chỉ số quan trọng tổng hợp = w_corr * |Pearson r| + w_pct * PctRankDev_norm
# Tổng trọng số phải bằng 1.0.

FEATURE_IMPORTANCE_WEIGHTS = {
    "pearson_abs_corr":  0.5,   # Trọng số tương quan Pearson tuyệt đối |r|
    "pct_rank_dev_norm": 0.5,   # Trọng số Median Percentile Rank Deviation chuẩn hóa
}

# Ngưỡng Stability Check (Spearman rho giữa 2 sub-sample anomaly):
STABILITY_RHO_THRESHOLD: float = 0.6   # rho < 0.6 → cảnh báo ranking không ổn định

# Phạm vi cực đoan dùng trong Co-dominant Feature Detection (RCA)
CO_DOMINANT_PCT_MIN:  float = 49.0     # Pct deviation >= 49.0% được coi là cực đoan
CO_DOMINANT_MARGIN:   float = 0.15     # Sai lệch tối đa so với max để vẫn là "đồng cực đoan"

# Hệ số pha trộn IQR tie-breaker (không để IQR át pct_dev chính)
IQR_TIEBREAK_SCALE:   float = 1e-4    # composite = pct_dev + min(iqr_dev, 1000) * 1e-4
IQR_TIEBREAK_CAP:     float = 1000.0  # Giới hạn trên cho iqr_dev trước khi nhân scale


# ==============================================================================
# 5. CẤU HÌNH DỮ LIỆU — NASA SHUTTLE TELEMETRY
# ==============================================================================
# UCI Statlog Shuttle: 10 cột = 9 cảm biến thực sự + 1 nhãn lớp (1-7).
# feat_10 là nhãn UCI — KHÔNG được đưa vào model khi huấn luyện.

N_SENSOR_COLS:  int = 9                                              # Số cột cảm biến thực sự
SENSOR_COLS         = [f"feat_{i+1}" for i in range(N_SENSOR_COLS)] # feat_1 → feat_9
LABEL_COL:  str = "feat_10"                                          # Cột nhãn UCI lớp 1-7

# Nhãn nhị phân cho External Validation:
# Class 1 (Rad Flow) = bình thường (78.6% dữ liệu); Class 2-7 = bất thường
NORMAL_CLASS: int = 1   # Giá trị nhãn UCI ứng với trạng thái bình thường


# ==============================================================================
# 6. HẰNG SỐ TOÁN HỌC CỐT LÕI — ISOLATION FOREST
# ==============================================================================

# Euler-Mascheroni constant gamma — dùng trong c_factor(n) (Liu et al., 2008)
EULER_MASCHERONI: float = 0.5772156649015329

# Giá trị max_samples mặc định khi "auto" (theo bài báo gốc Liu et al., 2008)
DEFAULT_MAX_SAMPLES_AUTO: int = 256

# Quy ước nhãn dự đoán (khác scikit-learn dùng -1/1)
LABEL_ANOMALY: int = 1    # Bất thường
LABEL_NORMAL:  int = 0    # Bình thường

# Ngưỡng offset mặc định (khi contamination="auto")
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_OFFSET:    float = -0.5


# ==============================================================================
# 7. CẤU HÌNH ĐÁNH GIÁ (EVALUATION)
# ==============================================================================

# Permutation Importance
PERMUTATION_N_REPEATS: int = 5          # Số lần hoán vị để ước lượng trung bình
PERMUTATION_SCORING:   str = "roc_auc"  # "roc_auc" hoặc "f1"

# Số mẫu bất thường tối thiểu để chạy Stability Check
STABILITY_MIN_ANOMALIES: int = 10

# Top-K mẫu bất thường nhất xuất trong báo cáo RCA
RCA_TOP_K: int = 10

# Khoảng chuẩn hóa PctRankDev: max có thể là 50.0 → [0, 50]
PCT_DEV_MAX_RANGE: float = 50.0


# ==============================================================================
# 8. CẤU HÌNH QUẢN LÝ FILE TRỌNG SỐ (WEIGHT FILES)
# ==============================================================================

BASELINE_JSON: str = "baseline_weights.json"
BASELINE_NPZ:  str = "baseline_weights.npz"
BASELINE_TXT:  str = "baseline_weights.txt"
MODEL_JSON:    str = "best_model_weights.json"

REQUIRED_WEIGHT_FILES = (BASELINE_JSON, BASELINE_NPZ, BASELINE_TXT, MODEL_JSON)


"""
WEIGHTS I/O — ISOLATION FOREST WEIGHT SERIALIZATION & CACHING
============================================================
Tệp xử lý lưu/nạp trọng số cho Baseline (Distance-to-Centroid)
và mô hình Isolation Forest tốt nhất.
Zero Scikit-Learn Dependency — Sử dụng json, numpy, os, hashlib, datetime.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from model import IsolationForest


def calculate_data_sha256(data_path: str) -> str:
    """Tính mã băm SHA-256 của file dữ liệu để kiểm tra tính toàn vẹn."""
    sha256_hash = hashlib.sha256()
    with open(data_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ==============================================================================
# 1. BASELINE MODEL (DISTANCE-TO-CENTROID)
# ==============================================================================

def fit_baseline(X_train: Any, sensor_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Huấn luyện mô hình Baseline (Distance-to-Centroid) trên tập TRAIN.
    Tính toán train_mean, train_std, dist_min, dist_max hoàn toàn trên tập TRAIN (chống rò rỉ tập test).
    """
    X_tr = X_train.values if hasattr(X_train, "values") else np.asarray(X_train, dtype=np.float64)
    train_mean = np.mean(X_tr, axis=0)
    train_std = np.std(X_tr, axis=0)
    train_std = np.where(train_std == 0.0, 1.0, train_std)  # Thay std = 0 thành 1.0

    # Chuẩn hóa tập TRAIN và tính khoảng cách Euclidean L2
    X_tr_norm = (X_tr - train_mean) / train_std
    train_dists = np.sqrt(np.sum(X_tr_norm ** 2, axis=1))

    dist_min = float(np.min(train_dists))
    dist_max = float(np.max(train_dists))

    cols = sensor_cols if sensor_cols is not None else [f"feat_{i+1}" for i in range(X_tr.shape[1])]

    return {
        "sensor_cols": cols,
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "dist_min": dist_min,
        "dist_max": dist_max,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def baseline_score(X: Any, w: Dict[str, Any]) -> np.ndarray:
    """
    Tính điểm số bất thường (Min-Max scaled [0, 1]) của mô hình Baseline.
    Sử dụng train_mean, train_std, dist_min, dist_max đã lưu từ tập TRAIN.
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float64)
    train_mean = np.array(w["train_mean"], dtype=np.float64)
    train_std = np.array(w["train_std"], dtype=np.float64)
    dist_min = float(w["dist_min"])
    dist_max = float(w["dist_max"])

    X_norm = (X_arr - train_mean) / train_std
    dists = np.sqrt(np.sum(X_norm ** 2, axis=1))

    # Min-max scaling theo dist_min / dist_max của tập TRAIN
    scores = (dists - dist_min) / (dist_max - dist_min + 1e-12)
    return scores


def save_baseline(w: Dict[str, Any], weights_dir: str = "weights") -> str:
    """Lưu trọng số baseline vào baseline_weights.json."""
    os.makedirs(weights_dir, exist_ok=True)
    filepath = os.path.join(weights_dir, "baseline_weights.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(w, f, indent=2, ensure_ascii=False)
    return filepath


def load_baseline(weights_dir: str = "weights") -> Dict[str, Any]:
    """Nạp trọng số baseline từ baseline_weights.json."""
    filepath = os.path.join(weights_dir, "baseline_weights.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file trọng số baseline: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 2. BEST ISOLATION FOREST MODEL
# ==============================================================================

def save_best_model(
    model: IsolationForest,
    best_params: Dict[str, Any],
    tuning: Optional[Dict[str, Any]],
    metrics: Dict[str, float],
    meta: Dict[str, Any],
    weights_dir: str = "weights"
) -> str:
    """
    Lưu bộ trọng số TỐT NHẤT của mô hình IsolationForest vào best_model_weights.json.
    Ghi JSON với ensure_ascii=False. Phần "trees" dạng gọn (compact), phần còn lại indent=2.
    """
    os.makedirs(weights_dir, exist_ok=True)
    filepath = os.path.join(weights_dir, "best_model_weights.json")

    model_dict = model.to_dict()
    trees = model_dict.get("trees", [])

    # Tạo bản sao model_dict và thay thế các phần tử trong "trees" bằng chuỗi placeholder
    placeholders = [f"__TREE_{i}__" for i in range(len(trees))]
    model_dict["trees"] = placeholders

    full_dict = {
        "meta": meta,
        "best_params": best_params,
        "tuning": tuning,
        "metrics": metrics,
        "model": model_dict
    }

    # Dump full_dict với indent=2
    json_str = json.dumps(full_dict, indent=2, ensure_ascii=False)

    # Thay thế từng placeholder bằng JSON compact của tree
    for i, tree in enumerate(trees):
        compact_tree_str = json.dumps(tree, ensure_ascii=False)
        json_str = json_str.replace(f'"__TREE_{i}__"', compact_tree_str)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json_str)

    return filepath


def load_best_model(weights_dir: str = "weights") -> Tuple[IsolationForest, Dict[str, Any]]:
    """
    Nạp mô hình tốt nhất từ best_model_weights.json.
    Trả về (best_model, full_data_dict).
    """
    filepath = os.path.join(weights_dir, "best_model_weights.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = IsolationForest.from_dict(data["model"])
    return model, data


# ==============================================================================
# 3. UTILITIES & CACHE VALIDATION
# ==============================================================================

def weights_exist(weights_dir: str = "weights") -> bool:
    """Kiểm tra xem cả 2 file trọng số có tồn tại và hợp lệ không."""
    baseline_path = os.path.join(weights_dir, "baseline_weights.json")
    model_path = os.path.join(weights_dir, "best_model_weights.json")
    return (
        os.path.exists(baseline_path) and os.path.getsize(baseline_path) > 0 and
        os.path.exists(model_path) and os.path.getsize(model_path) > 0
    )


def is_cache_valid(
    meta: Dict[str, Any],
    data_path: str,
    random_state: int,
    test_size: float,
    explicit_override: bool = False
) -> bool:
    """
    Kiểm tra tính hợp lệ của cache trọng số:
    - Nếu explicit_override=True (người dùng truyền tham số từ CLI): coi là cache không hợp lệ -> train lại.
    - So sánh SHA-256 của file dữ liệu + random_state + test_size.
    """
    if explicit_override:
        print("[!] Truyền tham số mô hình tường minh từ CLI -> Bỏ qua cache, huấn luyện lại.")
        return False

    if not os.path.exists(data_path):
        print(f"[!] Lỗi: File dữ liệu '{data_path}' không tồn tại!")
        return False

    current_sha256 = calculate_data_sha256(data_path)
    saved_sha256 = meta.get("data_sha256")
    saved_random_state = meta.get("random_state")
    saved_test_size = meta.get("test_size")

    if current_sha256 != saved_sha256:
        print(f"[!] Cảnh báo: SHA-256 dữ liệu không khớp ({current_sha256[:8]}... != {str(saved_sha256)[:8]}...) -> Train lại.")
        return False

    if saved_random_state != random_state:
        print(f"[!] Cảnh báo: random_state không khớp ({random_state} != {saved_random_state}) -> Train lại.")
        return False

    if saved_test_size is None or abs(float(saved_test_size) - float(test_size)) > 1e-6:
        print(f"[!] Cảnh báo: test_size không khớp ({test_size} != {saved_test_size}) -> Train lại.")
        return False

    return True

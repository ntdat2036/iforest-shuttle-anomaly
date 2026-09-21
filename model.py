"""
ISOLATION FOREST MODEL FROM SCRATCH (PURE NUMPY)
Chuẩn thuật toán Isolation Forest (Liu et al., 2008) - Zero Scikit-Learn Dependency.
"""

import math
from typing import List, Optional, Union
# pyrefly: ignore [missing-import]
import numpy as np


def c_factor(n: int) -> float:
    """Độ dài đường đi trung bình tìm kiếm không thành công trong BST (Liu et al., 2008)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    return 2.0 * (math.log(n - 1.0) + gamma) - (2.0 * (n - 1.0) / n)


class Node:
    """Nút cây cô lập: Nút trong (chia thuộc tính) hoặc Nút lá (lưu kích thước mẫu & leaf_adj)."""
    __slots__ = ['left', 'right', 'split_feat', 'split_val', 'size', 'is_leaf', 'leaf_adj']

    def __init__(self, left=None, right=None, split_feat=None, split_val=None, size: int = 0, is_leaf: bool = False):
        self.left = left
        self.right = right
        self.split_feat = split_feat
        self.split_val = split_val
        self.size = size
        self.is_leaf = is_leaf
        self.leaf_adj = c_factor(size) if is_leaf else 0.0


class IsolationTree:
    """Cây cô lập nhị phân xây dựng bằng phân hoạch ngẫu nhiên không gian (Algorithm 2, Liu et al., 2008)."""

    def __init__(self, max_depth: int, features_subset: Optional[np.ndarray] = None, rng: Optional[np.random.RandomState] = None, random_state: int = 42):
        self.max_depth = max_depth
        self.features_subset = features_subset
        if rng is not None:
            self.rng = rng
        elif random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray):
        self.root = self._build_tree(X, current_depth=0)
        return self

    def _build_tree(self, X: np.ndarray, current_depth: int) -> Node:
        n_samples, n_features = X.shape
        if current_depth >= self.max_depth or n_samples <= 1:
            return Node(size=n_samples, is_leaf=True)

        if self.features_subset is not None:
            sub_X = X[:, self.features_subset]
            col_mins = np.min(sub_X, axis=0)
            col_maxs = np.max(sub_X, axis=0)
            valid_idx = np.where(col_mins < col_maxs)[0]

            if len(valid_idx) == 0:
                return Node(size=n_samples, is_leaf=True)

            # Chọn ngẫu nhiên đều 1 thuộc tính từ tập các thuộc tính hợp lệ (Algorithm 2, Liu et al., 2008)
            pick = int(self.rng.choice(valid_idx))
            chosen_feat = int(self.features_subset[pick])
            feat_min, feat_max = float(col_mins[pick]), float(col_maxs[pick])
        else:
            col_mins = np.min(X, axis=0)
            col_maxs = np.max(X, axis=0)
            valid_idx = np.where(col_mins < col_maxs)[0]

            if len(valid_idx) == 0:
                return Node(size=n_samples, is_leaf=True)

            # Chọn ngẫu nhiên đều 1 thuộc tính từ toàn bộ không gian thuộc tính
            pick = int(self.rng.choice(valid_idx))
            chosen_feat = pick
            feat_min, feat_max = float(col_mins[pick]), float(col_maxs[pick])

        split_val = float(self.rng.uniform(feat_min, feat_max))
        left_mask = X[:, chosen_feat] < split_val
        n_left = int(np.count_nonzero(left_mask))

        if n_left == 0 or n_left == n_samples:
            return Node(size=n_samples, is_leaf=True)

        return Node(
            left=self._build_tree(X[left_mask], current_depth + 1),
            right=self._build_tree(X[~left_mask], current_depth + 1),
            split_feat=chosen_feat,
            split_val=split_val,
            size=n_samples,
            is_leaf=False
        )

    def compute_path_lengths(self, X: np.ndarray) -> np.ndarray:
        """Tính toán độ dài đường đi h(x) theo lô (Batch Traversal với Stack lặp tối ưu)."""
        n_samples = len(X)
        lengths = np.zeros(n_samples, dtype=np.float64)
        if self.root is None:
            return lengths

        stack = [(np.arange(n_samples), self.root, 0)]

        while stack:
            indices, curr_node, depth = stack.pop()
            if len(indices) == 0 or curr_node is None:
                continue
            if curr_node.is_leaf:
                lengths[indices] = depth + curr_node.leaf_adj
                continue
            if curr_node.split_feat is None or curr_node.split_val is None:
                continue
            col_vals = X[indices, curr_node.split_feat]
            mask = col_vals < curr_node.split_val
            n_left = int(np.count_nonzero(mask))
            if n_left < len(indices) and curr_node.right is not None:
                stack.append((indices[~mask], curr_node.right, depth + 1))
            if n_left > 0 and curr_node.left is not None:
                stack.append((indices[mask], curr_node.left, depth + 1))

        return lengths

    def to_dict(self) -> dict:
        """Flatten cây bằng duyệt tiền thứ tự (preorder traversal) thành mảng cùng độ dài."""
        if self.root is None:
            return {"feature": [], "threshold": [], "left": [], "right": [], "size": []}

        feature: List[int] = []
        threshold: List[Optional[float]] = []
        left: List[int] = []
        right: List[int] = []
        size: List[int] = []

        def _traverse(node: Node) -> int:
            idx = len(feature)
            feature.append(-1)
            threshold.append(None)
            left.append(-1)
            right.append(-1)
            size.append(node.size)

            if node.is_leaf:
                feature[idx] = -1
                threshold[idx] = None
                left[idx] = -1
                right[idx] = -1
            else:
                feature[idx] = int(node.split_feat) if node.split_feat is not None else -1
                threshold[idx] = float(node.split_val) if node.split_val is not None else None
                left[idx] = _traverse(node.left) if node.left is not None else -1
                right[idx] = _traverse(node.right) if node.right is not None else -1

            return idx

        _traverse(self.root)
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "size": size,
        }

    @classmethod
    def from_dict(cls, d: dict, max_depth: int) -> "IsolationTree":
        """Tái dựng lại IsolationTree từ dictionary."""
        tree = cls(max_depth=max_depth)
        features = d.get("feature", [])
        thresholds = d.get("threshold", [])
        lefts = d.get("left", [])
        rights = d.get("right", [])
        sizes = d.get("size", [])

        if not features:
            tree.root = None
            return tree

        def _build(idx: int) -> Optional[Node]:
            if idx < 0 or idx >= len(features):
                return None

            feat = features[idx]
            val = thresholds[idx]
            left_idx = lefts[idx]
            right_idx = rights[idx]
            sz = sizes[idx]

            is_leaf = (feat == -1 or feat is None or left_idx == -1)
            if is_leaf:
                # Nút lá: leaf_adj tự động tính lại bằng c_factor(sz)
                return Node(size=sz, is_leaf=True)

            left_node = _build(left_idx) if left_idx != -1 else None
            right_node = _build(right_idx) if right_idx != -1 else None

            return Node(
                left=left_node,
                right=right_node,
                split_feat=int(feat),
                split_val=float(val) if val is not None else None,
                size=int(sz),
                is_leaf=False
            )

        tree.root = _build(0)
        return tree



class IsolationForest:
    """
    Thuật toán Isolation Forest tự lập trình (Liu et al., 2008) - 100% Pure NumPy.
    
    Quy ước Nhãn & Hàm Mục Tiêu:
    -----------------------------
    - predict(): Trả về nhãn nhị phân chuẩn bài toán Anomaly Detection:
        + 1: BẤT THƯỜNG (Anomaly / Outlier)
        + 0: BÌNH THƯỜNG (Normal / Inlier)
      (Lưu ý: Khác với quy ước -1/1 của scikit-learn để khớp trực tiếp với biến mục tiêu {0, 1} của bài toán).
    - score_samples(X): Trả về -anomaly_score(X) (giá trị càng âm càng bất thường).
    - decision_function(X): Trả về threshold_ - anomaly_score(X) (điểm âm = bất thường, điểm dương = bình thường).
    
    Đặc điểm Kiến trúc & Mở rộng Kỹ thuật:
    --------------------------------------
    - Feature Bagging per Tree: Khi max_features < 1.0, tập thuộc tính con được chọn ngẫu nhiên
      1 lần cho mỗi cây iTree (theo thiết kế mở rộng phổ biến trong scikit-learn; bài báo gốc
      Liu et al. 2008 mặc định sử dụng toàn bộ thuộc tính cho mọi cây).
    - Input Validation: Phương thức _validate_X sử dụng np.isfinite để chặn toàn bộ dữ liệu chứa NaN/Inf,
      kiểm tra mảng 2 chiều, kích thước mẫu n >= 2 và số chiều thuộc tính khớp với tập huấn luyện.
    
    Tham số:
    ---------
    n_estimators : int, mặc định=100
        Số lượng cây cô lập iTree trong rừng.
    max_samples : int, float hoặc "auto", mặc định="auto"
        Số lượng mẫu rút trích để xây dựng mỗi cây (mặc định min(256, n_samples)).
    max_features : float, mặc định=1.0
        Tỷ lệ thuộc tính trích chọn ngẫu nhiên cho mỗi cây (Feature Bagging per Tree).
    contamination : float hoặc "auto", mặc định="auto"
        - "auto": Ngưỡng quyết định lý thuyết s >= 0.5 từ bài báo gốc Liu et al. (2008).
        - float trong (0, 0.5): Ước lượng ngưỡng phân vị từ điểm anomaly_score trên tập train.
    random_state : int, mặc định=42
        Khởi tạo seed số ngẫu nhiên đảm bảo tính tái lập (reproducibility).
    """

    def __init__(self, n_estimators: int = 100, max_samples: Union[int, float, str] = "auto", max_features: float = 1.0,
                 contamination: Union[float, str] = "auto", random_state: Union[int, np.random.RandomState, None] = 42):
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")

        if max_samples != "auto":
            if isinstance(max_samples, bool):
                raise ValueError("max_samples cannot be boolean")
            if not isinstance(max_samples, (int, float)) or max_samples <= 0:
                raise ValueError("max_samples must be > 0")
            if isinstance(max_samples, int) and max_samples < 2:
                raise ValueError("max_samples as integer must be >= 2")
            if isinstance(max_samples, float) and max_samples > 1.0:
                raise ValueError("max_samples as float must be in (0, 1]")

        if isinstance(max_features, bool) or not isinstance(max_features, (int, float)) or not (0 < max_features <= 1):
            raise ValueError("max_features must be in (0, 1]")

        if contamination != "auto":
            if isinstance(contamination, bool) or not isinstance(contamination, (int, float)) or not (0 < contamination < 0.5):
                raise ValueError("contamination must be in (0, 0.5)")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state

        # Initialise max_depth based on provided max_samples (used for immediate checks, will update in fit()).
        if self.max_samples == "auto":
            self.max_depth = int(math.ceil(math.log2(256)))
        elif isinstance(self.max_samples, int):
            self.max_depth = int(math.ceil(math.log2(self.max_samples)))
        else:
            self.max_depth = None

        self.trees: List[IsolationTree] = []
        self.max_samples_actual_ = 256
        self.c_psi_ = 1.0
        self.threshold_ = 0.5
        self.offset_ = -0.5
        self.n_features_in_: Optional[int] = None


    def _validate_X(self, X, check_features: bool = False) -> np.ndarray:
        """Kiểm tra và tiền xác thực định dạng dữ liệu đầu vào (NaN, Inf, số chiều)."""
        if hasattr(X, "values"):
            X_arr = np.asarray(X.values, dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(f"Dữ liệu đầu vào phải là mảng 2 chiều (n_samples, n_features), nhận được {X_arr.ndim} chiều.")

        if X_arr.shape[0] == 0:
            raise ValueError("Dữ liệu đầu vào không được rỗng (n_samples == 0).")

        if X_arr.shape[1] == 0:
            raise ValueError("Dữ liệu đầu vào không có thuộc tính nào (n_features == 0).")

        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Dữ liệu đầu vào chứa giá trị không hợp lệ (NaN hoặc Inf).")

        if check_features:
            if self.n_features_in_ is None:
                raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi 'fit(X)' trước.")
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Số lượng thuộc tính không khớp: mô hình được fit với {self.n_features_in_} thuộc tính, "
                    f"nhưng dữ liệu truyền vào có {X_arr.shape[1]} thuộc tính."
                )
        return X_arr

    @property
    def estimators_(self):
        """Danh sách các cây cô lập đã huấn luyện (Scikit-Learn convention)."""
        return self.trees

    def fit(self, X, y=None):
        """Huấn luyện tập hợp cây Isolation Trees trên dữ liệu X (Unsupervised, bỏ qua y)."""
        X_arr = self._validate_X(X, check_features=False)
        n_samples, n_features = X_arr.shape
        if n_samples < 2:
            raise ValueError(f"Dữ liệu huấn luyện cần ít nhất 2 mẫu để phân tách cô lập (n_samples >= 2), hiện có {n_samples} mẫu.")
        self.n_features_in_ = n_features
        if isinstance(self.random_state, np.random.RandomState):
            rng = self.random_state
        else:
            rng = np.random.RandomState(self.random_state)

        # 1. Xác định kích thước mẫu con psi (bảo đảm psi >= 2) và tính max_depth theo bài báo Liu et al. (2008)
        if self.max_samples == "auto":
            self.max_samples_actual_ = max(2, min(256, n_samples))
        elif isinstance(self.max_samples, float) and 0.0 < self.max_samples <= 1.0:
            frac_samples = int(round(self.max_samples * n_samples))
            self.max_samples_actual_ = max(2, min(frac_samples, n_samples))
        else:
            self.max_samples_actual_ = max(2, min(int(self.max_samples), n_samples))

        # max_depth dựa trên kích thước mẫu thực tế của mỗi cây psi = max_samples_actual_ (Liu et al., 2008: h_max = ceil(log2(psi)))
        self.max_depth = int(math.ceil(math.log2(self.max_samples_actual_)))
        self.c_psi_ = c_factor(self.max_samples_actual_)

        # 2. Xác định số thuộc tính trích chọn cho mỗi cây (Feature Bagging per Tree)
        k_feat = max(1, int(round(self.max_features * n_features)))

        self.trees = []
        for _ in range(self.n_estimators):
            sub_idx = rng.choice(n_samples, size=self.max_samples_actual_, replace=False)
            feat_subset = rng.choice(n_features, size=k_feat, replace=False) if k_feat < n_features else None
            tree = IsolationTree(max_depth=self.max_depth, features_subset=feat_subset, rng=rng)
            tree.fit(X_arr[sub_idx])
            self.trees.append(tree)

        # 3. Xác định ngưỡng theo chuẩn tổng quát (không hardcode bất kỳ dataset nào)
        if self.contamination == "auto":
            self.threshold_ = 0.5  # Ngưỡng lý thuyết tự nhiên bài báo gốc Liu et al. (2008)
            self.offset_ = -0.5
        elif isinstance(self.contamination, (int, float)):
            contam_val = float(self.contamination)
            if not (0.0 < contam_val < 0.5):
                raise ValueError(f"contamination phải nằm trong khoảng (0, 0.5), hiện là {contam_val}")
            train_scores = self.anomaly_score(X_arr)
            self.threshold_ = float(np.percentile(train_scores, 100.0 * (1.0 - contam_val)))
            self.offset_ = -self.threshold_
        else:
            raise ValueError(f"contamination không hợp lệ: {self.contamination}. Phải là 'auto' hoặc số thực trong khoảng (0, 0.5)")

        return self

    def anomaly_score(self, X) -> np.ndarray:
        """
        Tính toán Anomaly Score s(x, psi) = 2^(- E(h(x)) / c(psi)) theo Liu et al. (2008).
        Khoảng giá trị [0, 1]. Điểm càng CAO (tiến tới 1.0) càng BẤT THƯỜNG.
        """
        X_arr = self._validate_X(X, check_features=True)
        if not self.trees or self.c_psi_ == 0.0:
            return np.zeros(len(X_arr))

        path_lengths = np.zeros(len(X_arr), dtype=np.float64)
        for tree in self.trees:
            path_lengths += tree.compute_path_lengths(X_arr)

        avg_path_length = path_lengths / len(self.trees)
        return np.power(2.0, -avg_path_length / self.c_psi_)

    def score_samples(self, X) -> np.ndarray:
        """
        Trả về điểm đối nghịch: -anomaly_score(X).
        Quy ước chuẩn: Giá trị càng THẤP (càng âm) càng BẤT THƯỜNG.
        """
        return -self.anomaly_score(X)

    def decision_function(self, X) -> np.ndarray:
        """
        Hàm quyết định phân loại theo quy ước chuẩn:
        decision_function(X) = threshold_ - anomaly_score(X).
        - Điểm ÂM (< 0): BẤT THƯỜNG (Anomaly)
        - Điểm DƯƠNG (>= 0): BÌNH THƯỜNG (Normal)
        """
        return self.threshold_ - self.anomaly_score(X)

    def predict(self, X, threshold=None) -> np.ndarray:
        """
        Phân loại nhị phân:
        - 1: BẤT THƯỜNG (Anomaly)
        - 0: BÌNH THƯỜNG (Normal)
        """
        thr = self.threshold_ if threshold is None else float(threshold)
        scores = self.anomaly_score(X)
        return (scores >= thr).astype(int)

    def fit_predict(self, X, y=None, threshold=None) -> np.ndarray:
        """Vừa huấn luyện mô hình vừa trả về nhãn dự đoán (Unsupervised, bỏ qua y)."""
        return self.fit(X, y=y).predict(X, threshold=threshold)

    def to_dict(self) -> dict:
        """Xuất mô hình IsolationForest thành dict để serialize JSON."""
        rs_val = self.random_state if not isinstance(self.random_state, np.random.RandomState) else None
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "contamination": self.contamination,
                "random_state": rs_val,
            },
            "fitted": {
                "max_samples_actual_": self.max_samples_actual_,
                "max_depth": self.max_depth,
                "c_psi_": float(self.c_psi_),
                "threshold_": float(self.threshold_),
                "offset_": float(self.offset_),
                "n_features_in_": self.n_features_in_,
            },
            "trees": [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IsolationForest":
        """Dựng lại mô hình IsolationForest từ dict."""
        params = d.get("params", {})
        model = cls(
            n_estimators=params.get("n_estimators", 100),
            max_samples=params.get("max_samples", "auto"),
            max_features=params.get("max_features", 1.0),
            contamination=params.get("contamination", "auto"),
            random_state=params.get("random_state", 42),
        )

        fitted = d.get("fitted", {})
        model.max_samples_actual_ = fitted.get("max_samples_actual_", 256)
        model.max_depth = fitted.get("max_depth", 8)
        model.c_psi_ = float(fitted.get("c_psi_", 1.0))
        model.threshold_ = float(fitted.get("threshold_", 0.5))
        model.offset_ = float(fitted.get("offset_", -0.5))
        model.n_features_in_ = fitted.get("n_features_in_", None)

        trees_data = d.get("trees", [])
        model.trees = [IsolationTree.from_dict(t_dict, max_depth=model.max_depth) for t_dict in trees_data]
        return model


# Alias tương thích
IsolationForestScratch = IsolationForest


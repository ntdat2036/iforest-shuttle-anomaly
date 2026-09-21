
"""==============================================================================
UNIT TESTS & VERIFICATION SUITE FOR ISOLATION FOREST PIPELINE (ZERO SKLEARN)
=============================================================================="""

import os
import sys
import unittest
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

# Thêm thư mục gốc để import từ model và run_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import IsolationForest, IsolationForestScratch, c_factor
from run_pipeline import (
    DEFAULT_CONFIG,
    PARAM_GRID,
    Pipeline,
    PipelineScratch,
    StratifiedKFold,
    accuracy_score_scratch,
    precision_score_scratch,
    recall_score_scratch,
    f1_score_scratch,
    balanced_accuracy_score_scratch,
    roc_auc_score_scratch,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    permutation_importance_scratch,
    train_test_split,
    threshold_from_contamination,
    KFold,
    unsupervised_feature_importance,
    explain_anomalies_root_cause,
    robust_deviation,
)


class TestIsolationForestPipeline(unittest.TestCase):

    def test_01_data_files_exist(self):
        """Kiểm tra sự tồn tại của các file dữ liệu và mã nguồn chính."""
        self.assertTrue(os.path.exists("shuttle.csv"), "shuttle.csv phải tồn tại")
        self.assertTrue(os.path.exists("model.py"), "model.py phải tồn tại")
        self.assertTrue(os.path.exists("config.py"), "config.py phải tồn tại")
        self.assertTrue(os.path.exists("weights_io.py"), "weights_io.py phải tồn tại")
        self.assertTrue(os.path.exists("run_pipeline.py"), "run_pipeline.py phải tồn tại")

    def test_02_raw_unlabelled_data_integrity(self):
        """Kiểm tra tính toàn vẹn của dữ liệu telemetry NASA Shuttle thô không nhãn."""
        df = pd.read_csv("shuttle.csv", header=None)
        self.assertEqual(df.shape[0], 58000, f"Dữ liệu phải có đúng 58,000 dòng, hiện có {df.shape[0]}")
        self.assertEqual(df.shape[1], 10, f"Dữ liệu thô phải có đúng 10 cột đặc trưng, hiện có {df.shape[1]}")
        self.assertEqual(df.isnull().sum().sum(), 0, "Dữ liệu không được chứa giá trị NaN")
        self.assertTrue(all(np.issubdtype(dtype, np.integer) for dtype in df.dtypes), "Toàn bộ 10 cột phải là số nguyên")

    def test_03_pipeline_scratch_inference(self):
        """Kiểm tra pipeline tự xây dựng (Scratch) thuần iForest và khả năng suy luận (Inference)."""
        model = IsolationForestScratch(n_estimators=10, max_samples=64, random_state=42)
        pipe = PipelineScratch(model=model)
        
        # Fit đúng 9 cảm biến thực sự (cột 0-8), không lấy cột nhãn UCI (cột 9)
        df = pd.read_csv("shuttle.csv", header=None, nrows=100)
        pipe.fit(df.iloc[:, :9].values)  # 9 sensor cols only — production behavior
        
        # Mẫu dữ liệu giả lập 9 thuộc tính cảm biến
        dummy_sample = np.array([[50.0] * 9])
        score = float(pipe.anomaly_score(dummy_sample)[0])
        pred = int(pipe.predict(dummy_sample)[0])
        
        self.assertTrue(0.0 <= score <= 1.0, f"Anomaly score phải nằm trong đoạn [0, 1], hiện là {score}")
        self.assertIn(pred, [0, 1], "Dự đoán phải là nhãn nhị phân {0, 1}")


    def test_04_config_validity(self):
        """Kiểm tra tính hợp lệ của cấu hình mặc định và lưới siêu tham số (In-Code Config)."""
        self.assertIsInstance(DEFAULT_CONFIG, dict, "Cấu hình mặc định phải là dict")
        self.assertIn("random_state", DEFAULT_CONFIG)
        self.assertIn("n_estimators", DEFAULT_CONFIG)
        self.assertIsInstance(PARAM_GRID, list, "PARAM_GRID phải là danh sách (list)")
        self.assertGreater(len(PARAM_GRID), 0, "PARAM_GRID phải có ít nhất 1 cấu hình")

    def test_05_c_factor_math(self):
        """Kiểm tra tính chính xác của hàm c_factor(n) theo Liu et al. (2008)."""
        self.assertEqual(c_factor(1), 0.0)
        self.assertEqual(c_factor(2), 1.0)
        c_256 = c_factor(256)
        expected = 2.0 * (np.log(255) + 0.5772156649015329) - 2.0 * 255 / 256
        self.assertAlmostEqual(c_256, expected, places=5)

    def test_06_metrics_math_accuracy(self):
        """Kiểm tra độ chính xác toán học của các hàm Metric tự viết."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 0])
        self.assertAlmostEqual(accuracy_score_scratch(y_true, y_pred), 4/6)
        self.assertAlmostEqual(precision_score_scratch(y_true, y_pred), 1/2)
        self.assertAlmostEqual(recall_score_scratch(y_true, y_pred), 1/2)
        self.assertAlmostEqual(balanced_accuracy_score_scratch(y_true, y_pred), (3/4 + 1/2)/2)
        # Test ROC-AUC hoàn hảo và test xử lý ties
        scores_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc_score_scratch(y_true, scores_perfect), 1.0)
        scores_ties = np.array([0.2, 0.2, 0.5, 0.5, 0.8, 0.8])
        auc_ties = roc_auc_score_scratch(y_true, scores_ties)
        self.assertTrue(0.0 <= auc_ties <= 1.0)

    def test_07_model_generalization_and_validation(self):
        """Kiểm tra tính tổng quát của contamination='auto', kiểm tra số chiều và bẫy lỗi NaN/Inf."""
        X_dummy = np.random.RandomState(42).randn(100, 4)
        model = IsolationForest(n_estimators=10, max_samples=32, contamination="auto", random_state=42)
        model.fit(X_dummy)
        
        # 1. contamination='auto' phải dùng ngưỡng lý thuyết 0.5 (không hardcode dataset nào)
        self.assertEqual(model.threshold_, 0.5)
        self.assertEqual(model.n_features_in_, 4)

        # 2. Kiểm tra score_samples, decision_function, fit_predict, estimators_
        self.assertEqual(len(model.estimators_), 10)
        scores = model.anomaly_score(X_dummy)
        self.assertTrue(np.allclose(model.score_samples(X_dummy), -scores))
        self.assertTrue(np.allclose(model.decision_function(X_dummy), 0.5 - scores))
        preds = model.fit_predict(X_dummy, y=np.zeros(100))
        self.assertEqual(len(preds), 100)

        # 3. Kiểm tra bẫy lỗi khi sai số chiều thuộc tính
        with self.assertRaises(ValueError):
            model.predict(np.random.randn(10, 5))

        # 4. Kiểm tra bẫy lỗi khi dữ liệu chứa NaN
        X_nan = X_dummy.copy()
        X_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.predict(X_nan)

    def test_08_curves_and_importance(self):
        """Kiểm tra tính toán đường cong ROC, PR, Average Precision và Permutation Feature Importance."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        # 1. Đường cong ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        self.assertEqual(len(fpr), len(tpr))
        self.assertTrue(np.all(fpr >= 0.0) and np.all(fpr <= 1.0))
        self.assertTrue(np.all(tpr >= 0.0) and np.all(tpr <= 1.0))

        # 2. Đường cong Precision-Recall & Average Precision
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        self.assertTrue(0.0 <= ap <= 1.0, f"AP phải nằm trong [0, 1], hiện là {ap}")

        # 3. Permutation Feature Importance
        X_dummy = np.random.RandomState(42).randn(100, 3)
        y_dummy = (X_dummy[:, 0] > 0.5).astype(int)
        model = IsolationForest(n_estimators=10, max_samples=32, contamination=0.2, random_state=42)
        model.fit(X_dummy)
        imp_mean, imp_std = permutation_importance_scratch(model, X_dummy, y_dummy, n_repeats=2, random_state=42)
        self.assertEqual(len(imp_mean), 3)
        self.assertEqual(len(imp_std), 3)

    def test_09_train_test_split_size_and_disjoint(self):
        """Kiểm tra phân chia tập dữ liệu 80/20 không giám sát: Train=46,400, Test=11,600 và không trùng lặp index."""
        df = pd.read_csv("shuttle.csv", header=None)
        X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), 46400, f"Train phải có đúng 46,400 mẫu, hiện là {len(X_train)}")
        self.assertEqual(len(X_test), 11600, f"Test phải có đúng 11,600 mẫu, hiện là {len(X_test)}")
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        self.assertTrue(train_indices.isdisjoint(test_indices), "Data overlap: train/test indices overlap!")

        # Kiểm tra thêm nhánh phân chia có nhãn (supervised fallback) trên dữ liệu giả lập
        X_dummy = np.random.RandomState(42).randn(100, 4)
        y_dummy = np.array([0] * 80 + [1] * 20)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=True
        )
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_te), 20)

    def test_10_iforest_config(self):
        """Kiểm tra cấu hình mô hình IsolationForest."""
        model = IsolationForest(
            n_estimators=100,
            max_samples=256,
            max_features=1.0,
            contamination="auto",
            random_state=42
        )
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.max_samples, 256)
        self.assertEqual(model.max_features, 1.0)
        self.assertEqual(model.contamination, "auto")

    def test_11_max_depth(self):
        """Kiểm tra max_depth = ceil(log2(max_samples)) = 8 với max_samples=256."""
        model = IsolationForest(
            n_estimators=100,
            max_samples=256,
            random_state=42
        )
        self.assertEqual(model.max_depth, 8)

    def test_12_c_factor_256(self):
        """Kiểm tra c_factor(256) tiệm cận chính xác 10.2447709201."""
        value = c_factor(256)
        self.assertTrue(np.isclose(value, 10.2447709201, atol=1e-8))

    def test_13_anomaly_score_range(self):
        """Kiểm tra anomaly score nằm trong miền [0, 1] và là số hữu hạn (finite)."""
        X_dummy = np.random.RandomState(42).randn(100, 4)
        model = IsolationForest(n_estimators=20, max_samples=64, random_state=42)
        model.fit(X_dummy)
        scores = model.anomaly_score(X_dummy)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_14_score_consistency(self):
        """Kiểm tra tính nhất quán toán học giữa anomaly_score, score_samples và decision_function."""
        X_dummy = np.random.RandomState(42).randn(100, 4)
        model = IsolationForest(n_estimators=20, max_samples=64, random_state=42)
        model.fit(X_dummy)
        anomaly_scores = model.anomaly_score(X_dummy)
        sample_scores = model.score_samples(X_dummy)
        decision = model.decision_function(X_dummy)
        self.assertTrue(np.allclose(sample_scores, -anomaly_scores))
        self.assertTrue(np.allclose(decision, 0.5 - anomaly_scores))
    def test_15_hyperparameter_validation(self):
        """Kiểm tra bẫy lỗi các siêu tham số không hợp lệ."""
        with self.assertRaises(ValueError):
            IsolationForest(n_estimators=0)
        with self.assertRaises(ValueError):
            IsolationForest(n_estimators=-10)
        with self.assertRaises(ValueError):
            IsolationForest(max_samples=0)
        with self.assertRaises(ValueError):
            IsolationForest(max_samples=1)  # Cần ít nhất 2 mẫu để phân tách cô lập
        with self.assertRaises(ValueError):
            IsolationForest(max_samples=-5)
        with self.assertRaises(ValueError):
            IsolationForest(max_features=2.0)
        with self.assertRaises(ValueError):
            IsolationForest(max_features=0.0)
        with self.assertRaises(ValueError):
            IsolationForest(contamination=0.9)
        with self.assertRaises(ValueError):
            IsolationForest(contamination=-0.1)
        with self.assertRaises(ValueError):
            train_test_split(np.zeros((10, 2)), np.zeros(10), test_size=1.5)
        with self.assertRaises(ValueError):
            train_test_split(np.zeros((10, 2)), np.zeros(8))
        with self.assertRaises(ValueError):
            train_test_split(np.zeros((1, 2)), np.zeros(1))
        with self.assertRaises(ValueError):
            train_test_split(np.zeros(10), np.zeros(10))
        with self.assertRaises(ValueError):
            threshold_from_contamination([0.1, 0.2], contamination=0.8)

    def test_16_edge_cases_and_pipeline_properties(self):
        """Kiểm tra các trường hợp biên, tính toán max_depth và xác thực StratifiedKFold."""
        # 1. Pipeline default init and properties
        pipe = Pipeline()
        self.assertIsNotNone(pipe.model)
        self.assertEqual(pipe.threshold_, 0.5)
        self.assertEqual(pipe.offset_, -0.5)
        self.assertEqual(pipe.max_depth, 8)

        # 2. Fit với RandomState object và kiểm tra max_depth tính từ max_samples_actual_
        rng = np.random.RandomState(123)
        model = IsolationForest(n_estimators=5, max_samples=32, random_state=rng)
        X_dummy = np.random.RandomState(42).randn(50, 4)
        model.fit(X_dummy)
        self.assertEqual(len(model.estimators_), 5)
        self.assertEqual(model.max_samples_actual_, 32)
        self.assertEqual(model.max_depth, 5)  # ceil(log2(32)) = 5

        # Khi n_samples < max_samples, max_depth phải co theo max_samples_actual_
        small_model = IsolationForest(n_estimators=5, max_samples=256, random_state=42)
        small_model.fit(X_dummy)  # n_samples = 50
        self.assertEqual(small_model.max_samples_actual_, 50)
        self.assertEqual(small_model.max_depth, 6)  # ceil(log2(50)) = 6

        # 3. Dữ liệu rỗng, n_samples < 2 hoặc sai số chiều
        with self.assertRaises(ValueError):
            IsolationForest().fit(np.zeros((0, 4)))
        with self.assertRaises(ValueError):
            IsolationForest().fit(np.zeros((1, 4)))  # Cần ít nhất 2 mẫu
        with self.assertRaises(ValueError):
            IsolationForest().fit(np.zeros((10, 0)))

        # 4. StratifiedKFold validation toàn diện
        with self.assertRaises(ValueError):
            StratifiedKFold(n_splits=1)
        skf = StratifiedKFold(n_splits=3)
        with self.assertRaises(ValueError):
            list(skf.split(np.zeros((10, 2)), np.zeros(8)))  # Lệch độ dài X và y
        with self.assertRaises(ValueError):
            list(skf.split(np.zeros((2, 2)), [0, 1]))  # n_samples < n_splits
        with self.assertRaises(ValueError):
            list(skf.split(np.zeros((6, 2)), [0, 0, 0, 0, 0, 0]))  # Chỉ 1 lớp
        with self.assertRaises(ValueError):
            list(skf.split(np.zeros((6, 2)), [0, 0, 0, 0, 0, 1]))  # Lớp thiểu số < n_splits

        # 5. Threshold from contamination edge cases
        with self.assertRaises(ValueError):
            threshold_from_contamination([], 0.1)
        with self.assertRaises(ValueError):
            threshold_from_contamination([0.1, 0.2], contamination="invalid")

        # 6. Metrics length mismatch
        with self.assertRaises(ValueError):
            confusion_matrix([0, 1], [0])
        with self.assertRaises(ValueError):
            accuracy_score_scratch([0, 1], [0])
        with self.assertRaises(ValueError):
            balanced_accuracy_score_scratch([0, 1], [0])
        with self.assertRaises(ValueError):
            precision_score_scratch([0, 1], [0])
        with self.assertRaises(ValueError):
            recall_score_scratch([0, 1], [0])
        with self.assertRaises(ValueError):
            f1_score_scratch([0, 1], [0])
        with self.assertRaises(ValueError):
            roc_auc_score_scratch([0, 1], [0.5])
        with self.assertRaises(ValueError):
            average_precision_score([0, 1], [0.5])
        with self.assertRaises(ValueError):
            permutation_importance_scratch(model, X_dummy, np.zeros(len(X_dummy)), n_repeats=0)
        with self.assertRaises(ValueError):
            permutation_importance_scratch(model, X_dummy, np.zeros(len(X_dummy)), scoring="invalid")

    def test_17_unsupervised_kfold(self):
        """Kiểm tra K-Fold không giám sát (Unsupervised K-Fold CV)."""
        # Kiểm tra validation tham số
        with self.assertRaises(ValueError):
            KFold(n_splits=1)
        with self.assertRaises(ValueError):
            list(KFold(n_splits=5).split(np.zeros((3, 2))))
        with self.assertRaises(ValueError):
            list(KFold(n_splits=3).split(None))

        # Kiểm tra phân chia chính xác n_splits và không giao nhau giữa train/val
        X_dummy = np.random.RandomState(42).randn(90, 4)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(kf.split(X_dummy))
        self.assertEqual(len(splits), 3)

        all_val_indices = []
        for train_idx, val_idx in splits:
            self.assertEqual(len(train_idx) + len(val_idx), 90)
            self.assertTrue(set(train_idx).isdisjoint(set(val_idx)), "Train và Val index phải rời rạc")
            all_val_indices.extend(val_idx)

        self.assertEqual(sorted(all_val_indices), list(range(90)), "Tập hợp các validation fold phải bao phủ toàn bộ dữ liệu")

    def test_18_unsupervised_feature_importance_and_root_cause(self):
        """Kiểm tra tính toán Feature Importance và Root Cause Analysis không giám sát."""
        X_dummy = np.random.RandomState(42).randn(100, 5)
        # Giả lập đặc trưng 0 có ngoại lệ cực lớn -> anomaly score sẽ tương quan mạnh với đặc trưng 0
        X_dummy[:5, 0] += 20.0
        model = IsolationForest(n_estimators=20, max_samples=64, random_state=42)
        model.fit(X_dummy)

        # 1. Feature Importance
        df_imp = unsupervised_feature_importance(model, X_dummy, feature_names=[f"sensor_{i}" for i in range(5)])
        self.assertEqual(len(df_imp), 5)
        self.assertIn("Đặc trưng", df_imp.columns)
        self.assertIn("Tương quan Pearson |r|", df_imp.columns)
        self.assertIn("Chỉ số quan trọng tổng hợp", df_imp.columns)
        self.assertTrue(np.all(df_imp["Chỉ số quan trọng tổng hợp"] >= 0))

        # 2. Root Cause Analysis
        df_rca = explain_anomalies_root_cause(model, X_dummy, top_k=5, feature_names=[f"sensor_{i}" for i in range(5)])
        self.assertEqual(len(df_rca), 5)
        self.assertIn("Anomaly Score", df_rca.columns)
        self.assertIn("Đặc trưng lệch mạnh nhất", df_rca.columns)
        self.assertIn("Mức cảnh báo", df_rca.columns)

    def test_19_unsupervised_pipeline_reproducibility(self):
        """Kiểm tra tính tái lập (Reproducibility) của pipeline với fixed random_state."""
        X_sub = pd.read_csv("shuttle.csv", header=None).iloc[:200].values
        m1 = IsolationForest(n_estimators=30, max_samples=128, random_state=42).fit(X_sub)
        m2 = IsolationForest(n_estimators=30, max_samples=128, random_state=42).fit(X_sub)
        scores1 = m1.anomaly_score(X_sub)
        scores2 = m2.anomaly_score(X_sub)
        np.testing.assert_allclose(scores1, scores2, atol=1e-9,
                                   err_msg="Hai mô hình cùng random_state phải cho Anomaly Score đồng nhất 100%")

    def test_20_no_label_leakage(self):
        """Xác nhận cột 10 (nhãn UCI) KHÔNG nằm trong feature matrix X dùng để huấn luyện."""
        df = pd.read_csv("shuttle.csv", header=None)
        # Cột 9 (0-indexed) là nhãn lớp: chỉ nhận giá trị {1,2,3,4,5,6,7}
        label_col = df.iloc[:, 9]
        unique_labels = set(label_col.unique())
        self.assertTrue(unique_labels.issubset({1, 2, 3, 4, 5, 6, 7}), "Cột 9 phải là nhãn lớp UCI {1..7}")
        self.assertEqual(len(unique_labels), 7, "UCI Shuttle phải có đủ 7 lớp")

        # Feature matrix X cho 9 cảm biến thực thụ
        X = df.iloc[:, :9]
        self.assertEqual(X.shape[1], 9, "X dùng để train phải có đúng 9 cột cảm biến, không bao gồm nhãn")

        # Kiểm tra không cột nào trong 9 cảm biến là nhãn lớp
        for col in range(9):
            vals = set(df.iloc[:, col].unique())
            self.assertFalse(
                vals.issubset({1, 2, 3, 4, 5, 6, 7}) and len(vals) <= 7,
                f"Cột {col} trông giống nhãn lớp, có nguy cơ nhầm lẫn"
            )

        # Kiểm tra train_test_split tách nhãn rời rạc
        y_raw = df.iloc[:, 9].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_raw, test_size=0.2, random_state=42)
        self.assertEqual(X_tr.shape[1], 9)
        self.assertEqual(X_te.shape[1], 9)
        self.assertEqual(len(y_tr), len(X_tr))
        self.assertEqual(len(y_te), len(X_te))

    def test_21_rca_stability(self):
        """Kiểm tra RCA ranking ổn định với robust_deviation phi tham số và chịu lỗi kurtosis cực cao."""
        # 1. Test robust_deviation trên dải đều
        col_vals = np.arange(100, dtype=float)
        dev_median = robust_deviation(50.0, col_vals)
        dev_extreme = robust_deviation(99.0, col_vals)
        self.assertLess(dev_median, dev_extreme, "Giá trị trung vị phải có độ lệch nhỏ hơn giá trị cực đoan")
        self.assertLessEqual(dev_extreme, 50.0, "Độ lệch phân vị tối đa là 50.0")

        # 2. Test với dữ liệu zero-inflated (tương tự feat_2, feat_4, feat_6 của shuttle)
        rng = np.random.RandomState(42)
        zero_inflated = np.concatenate([np.zeros(700), rng.randn(300) * 10])
        dev_zero = robust_deviation(0.0, zero_inflated)
        dev_outlier = robust_deviation(float(np.max(zero_inflated)), zero_inflated)
        self.assertLess(dev_zero, dev_outlier, "Outlier phải có percentile deviation lớn hơn giá trị 0 phổ biến")

        # 3. Test tính ổn định RCA trên mô hình thật
        X_dummy = np.random.RandomState(42).randn(120, 5)
        X_dummy[0, 0] = 100.0  # Mẫu 0 có feat_1 là outlier cực đoan duy nhất
        model = IsolationForest(n_estimators=25, max_samples=64, random_state=42).fit(X_dummy)

        df_rca = explain_anomalies_root_cause(model, X_dummy, top_k=5)
        self.assertEqual(len(df_rca), 5)
        self.assertIn("Pct Rank Deviation", df_rca.columns)
        self.assertIn("Pct rank trong quần thể", df_rca.columns)

        # Mẫu 0 phải đứng hạng 1 và feat_1 là root cause
        self.assertEqual(int(df_rca.iloc[0]["Chỉ số mẫu (Index)"]), 0)
        self.assertEqual(df_rca.iloc[0]["Đặc trưng lệch mạnh nhất"], "feat_1")

        # 4. Test feature importance stability check
        rng = np.random.RandomState(42)
        X_norm = rng.randn(150, 5)
        X_anom = rng.randn(30, 5)
        X_anom[:, 0] += 12.0
        X_anom[:, 1] += 8.0
        X_test_data = np.vstack([X_norm, X_anom])
        model2 = IsolationForest(n_estimators=30, max_samples=128, random_state=42).fit(X_test_data)
        df_imp = unsupervised_feature_importance(model2, X_test_data)
        self.assertIn("Độ lệch Pct-Rank (Dị biệt)", df_imp.columns)
        self.assertIn("stability_note", df_imp.attrs)
        self.assertIn("[OK]", df_imp.attrs["stability_note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)



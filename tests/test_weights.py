"""
UNIT TESTS FOR WEIGHT SERIALIZATION & IO (ZERO SKLEARN)
======================================================
1. Round-trip model qua JSON: anomaly_score và predict giống hệt.
2. Round-trip baseline: baseline_score giống hệt.
3. dist_min / dist_max của baseline không đổi khi thay X_test.
4. is_cache_valid = False khi đổi random_state, đổi nội dung dữ liệu, hoặc truyền CLI.
5. save_best_model và load_best_model qua file JSON thật.
6. save_best_model khi tuning=None.
7. Nạp từ thư mục rỗng raise FileNotFoundError.
8. Tính nhất quán của 3 file baseline (.json, .npz, .txt).
9. weights_exist yêu cầu đủ 4 file và size > 0.
10. sync_baseline_files tự động sinh/đồng bộ .npz và .txt từ JSON.
"""

import os
import sys
import tempfile
import json
import unittest
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import IsolationForest
from weights_io import (
    fit_baseline,
    baseline_score,
    save_baseline,
    load_baseline,
    load_baseline_npz,
    save_best_model,
    load_best_model,
    weights_exist,
    missing_weight_files,
    sync_baseline_files,
    is_cache_valid,
    calculate_data_sha256,
)


class TestWeightsSerialization(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.X_train = self.rng.randn(100, 5)
        self.X_test = self.rng.randn(30, 5)

        # Train a small model
        self.model = IsolationForest(n_estimators=10, max_samples=32, contamination=0.1, random_state=42)
        self.model.fit(self.X_train)

    def test_01_model_json_roundtrip(self):
        """1. Round-trip model qua JSON: anomaly_score và predict giống hệt (np.allclose)."""
        model_dict = self.model.to_dict()

        # Reconstruct model from dict
        reconstructed_model = IsolationForest.from_dict(model_dict)

        scores_orig = self.model.anomaly_score(self.X_test)
        scores_recon = reconstructed_model.anomaly_score(self.X_test)

        preds_orig = self.model.predict(self.X_test)
        preds_recon = reconstructed_model.predict(self.X_test)

        # Assert anomaly scores and predictions match 100%
        np.testing.assert_allclose(
            scores_orig, scores_recon, atol=1e-12,
            err_msg="Anomaly score của mô hình dựng lại từ JSON phải khớp 100% với mô hình gốc!"
        )
        np.testing.assert_array_equal(
            preds_orig, preds_recon,
            err_msg="Dự đoán nhãn nhị phân predict() phải khớp 100%!"
        )

        # Test leaf node threshold is None (serialized to null in JSON, not NaN)
        json_str = json.dumps(model_dict)
        self.assertNotIn("NaN", json_str, "JSON serialised không được chứa NaN!")

    def test_02_baseline_roundtrip(self):
        """2. Round-trip baseline: baseline_score giống hệt."""
        baseline_w = fit_baseline(self.X_train, sensor_cols=[f"feat_{i+1}" for i in range(5)])
        scores_orig = baseline_score(self.X_test, baseline_w)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_baseline(baseline_w, tmpdir)
            loaded_w = load_baseline(tmpdir)
            scores_loaded = baseline_score(self.X_test, loaded_w)

            np.testing.assert_allclose(
                scores_orig, scores_loaded, atol=1e-12,
                err_msg="Baseline score nạp từ file JSON phải khớp 100% với baseline score gốc!"
            )

    def test_03_baseline_dist_min_max_stability(self):
        """3. dist_min / dist_max của baseline không đổi khi thay X_test."""
        baseline_w = fit_baseline(self.X_train, sensor_cols=[f"feat_{i+1}" for i in range(5)])

        orig_dist_min = baseline_w["dist_min"]
        orig_dist_max = baseline_w["dist_max"]

        # Compute scores on 2 different test sets
        X_test_1 = self.rng.randn(20, 5)
        X_test_2 = self.rng.randn(50, 5) + 10.0  # extreme test set

        _ = baseline_score(X_test_1, baseline_w)
        _ = baseline_score(X_test_2, baseline_w)

        # Confirm dist_min and dist_max remain completely unchanged
        self.assertEqual(baseline_w["dist_min"], orig_dist_min)
        self.assertEqual(baseline_w["dist_max"], orig_dist_max)

    def test_04_cache_validity(self):
        """4. is_cache_valid = False khi đổi random_state, đổi nội dung dữ liệu, hoặc truyền CLI parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write dummy data file
            data_path = os.path.join(tmpdir, "dummy_data.csv")
            with open(data_path, "w") as f:
                f.write("1,2,3,4,5\n6,7,8,9,10\n")

            sha256 = calculate_data_sha256(data_path)
            meta = {
                "data_path": data_path,
                "data_sha256": sha256,
                "random_state": 42,
                "test_size": 0.2,
                "n_train": 80,
            }

            # Valid cache check
            self.assertTrue(is_cache_valid(meta, data_path, random_state=42, test_size=0.2, explicit_override=False))

            # Invalid: random_state changed
            self.assertFalse(is_cache_valid(meta, data_path, random_state=99, test_size=0.2, explicit_override=False))

            # Invalid: explicit_override=True (CLI parameters provided)
            self.assertFalse(is_cache_valid(meta, data_path, random_state=42, test_size=0.2, explicit_override=True))

            # Invalid: test_size changed
            self.assertFalse(is_cache_valid(meta, data_path, random_state=42, test_size=0.3, explicit_override=False))

            # Invalid: file data changed
            with open(data_path, "w") as f:
                f.write("modified content!\n")

            self.assertFalse(is_cache_valid(meta, data_path, random_state=42, test_size=0.2, explicit_override=False))

    def test_05_save_load_best_model_file_roundtrip(self):
        """5. save_best_model và load_best_model qua file JSON thật (kiểm tra compact tree, placeholders, round-trip)."""
        model = IsolationForest(n_estimators=12, max_samples=32, contamination=0.1, random_state=42)
        model.fit(self.X_train)

        best_params = {"n_estimators": 12, "max_samples": 32, "max_features": 1.0}
        tuning = {"criterion": "score_spread", "results": [], "best_index": 0}
        metrics = {"roc_auc": 0.85, "average_precision": 0.65, "score_spread": 0.07}
        meta = {"data_path": "shuttle.csv", "data_sha256": "12345", "random_state": 42, "test_size": 0.2, "n_train": 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_best_model(model, best_params, tuning, metrics, meta, tmpdir)
            expected_path = os.path.join(tmpdir, "best_model_weights.json")
            self.assertEqual(path, expected_path)
            self.assertTrue(os.path.exists(path) and os.path.getsize(path) > 0)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertNotIn("__TREE_", content)
            self.assertNotIn("NaN", content)

            data = json.loads(content)
            self.assertEqual(set(data.keys()), {"meta", "best_params", "tuning", "metrics", "model"})
            self.assertEqual(len(data["model"]["trees"]), 12)

            trees_dict = model.to_dict()["trees"]
            for idx in [0, 1, 10, 11]:
                compact_str = json.dumps(trees_dict[idx], ensure_ascii=False)
                self.assertIn(compact_str, content)

            model2, loaded_data = load_best_model(tmpdir)
            scores1 = model.anomaly_score(self.X_test)
            scores2 = model2.anomaly_score(self.X_test)
            np.testing.assert_allclose(scores1, scores2, atol=1e-12)

            preds1 = model.predict(self.X_test)
            preds2 = model2.predict(self.X_test)
            np.testing.assert_array_equal(preds1, preds2)

            self.assertEqual(loaded_data["best_params"], best_params)
            self.assertEqual(loaded_data["metrics"], metrics)
            self.assertEqual(loaded_data["meta"], meta)
            self.assertEqual(loaded_data["tuning"], tuning)

            self.assertEqual(model2.n_estimators, model.n_estimators)
            self.assertEqual(model2.max_depth, model.max_depth)
            self.assertEqual(model2.threshold_, model.threshold_)
            self.assertEqual(model2.offset_, model.offset_)

    def test_06_save_best_model_tuning_none(self):
        """6. save_best_model khi tuning=None -> lưu và nạp được, data['tuning'] is None, điểm số vẫn khớp."""
        model = IsolationForest(n_estimators=10, max_samples=32, contamination=0.1, random_state=42)
        model.fit(self.X_train)

        best_params = {"n_estimators": 10, "max_samples": 32, "max_features": 1.0}
        metrics = {"roc_auc": 0.85, "average_precision": 0.65, "score_spread": 0.07}
        meta = {"data_path": "shuttle.csv", "data_sha256": "12345", "random_state": 42, "test_size": 0.2, "n_train": 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_best_model(model, best_params, None, metrics, meta, tmpdir)
            model2, data = load_best_model(tmpdir)

            self.assertIsNone(data["tuning"])
            scores1 = model.anomaly_score(self.X_test)
            scores2 = model2.anomaly_score(self.X_test)
            np.testing.assert_allclose(scores1, scores2, atol=1e-12)

    def test_07_load_missing_files(self):
        """7. Nạp từ thư mục rỗng phải raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_best_model(tmpdir)
            with self.assertRaises(FileNotFoundError):
                load_baseline(tmpdir)
            with self.assertRaises(FileNotFoundError):
                load_baseline_npz(tmpdir)

    def test_08_baseline_three_files_consistent(self):
        """8. Kiểm tra tính nhất quán của 3 file baseline (.json, .npz, .txt)."""
        sensor_cols = [f"feat_{i+1}" for i in range(5)]
        baseline_w = fit_baseline(self.X_train, sensor_cols=sensor_cols)
        scores_orig = baseline_score(self.X_test, baseline_w)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_baseline(baseline_w, tmpdir)
            json_p = os.path.join(tmpdir, "baseline_weights.json")
            npz_p = os.path.join(tmpdir, "baseline_weights.npz")
            txt_p = os.path.join(tmpdir, "baseline_weights.txt")

            self.assertTrue(os.path.exists(json_p) and os.path.getsize(json_p) > 0)
            self.assertTrue(os.path.exists(npz_p) and os.path.getsize(npz_p) > 0)
            self.assertTrue(os.path.exists(txt_p) and os.path.getsize(txt_p) > 0)

            loaded_json = load_baseline(tmpdir)
            loaded_npz = load_baseline_npz(tmpdir)

            np.testing.assert_allclose(loaded_npz["train_mean"], loaded_json["train_mean"])
            np.testing.assert_allclose(loaded_npz["train_std"], loaded_json["train_std"])
            self.assertEqual(loaded_npz["dist_min"], loaded_json["dist_min"])
            self.assertEqual(loaded_npz["dist_max"], loaded_json["dist_max"])

            scores_npz = baseline_score(self.X_test, loaded_npz)
            np.testing.assert_allclose(scores_orig, scores_npz, atol=1e-12)

            with open(txt_p, "r", encoding="utf-8") as f:
                txt_content = f.read()

            self.assertIn("dist_min", txt_content)
            self.assertIn("dist_max", txt_content)
            for col in sensor_cols:
                self.assertIn(col, txt_content)
            fmt_mean0 = format(float(baseline_w["train_mean"][0]), ".17g")
            self.assertIn(fmt_mean0, txt_content)

    def test_09_weights_exist_requires_all_four(self):
        """9. weights_exist yêu cầu đủ cả 4 file và kích thước > 0."""
        sensor_cols = [f"feat_{i+1}" for i in range(5)]
        baseline_w = fit_baseline(self.X_train, sensor_cols=sensor_cols)
        best_params = {"n_estimators": 10, "max_samples": 32, "max_features": 1.0}
        metrics = {"roc_auc": 0.85, "average_precision": 0.65, "score_spread": 0.07}
        meta = {"data_path": "shuttle.csv", "data_sha256": "12345", "random_state": 42, "test_size": 0.2, "n_train": 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_baseline(baseline_w, tmpdir)
            save_best_model(self.model, best_params, None, metrics, meta, tmpdir)

            self.assertTrue(weights_exist(tmpdir))
            self.assertEqual(missing_weight_files(tmpdir), [])

            all_files = ["baseline_weights.json", "baseline_weights.npz", "baseline_weights.txt", "best_model_weights.json"]
            for f_name in all_files:
                f_path = os.path.join(tmpdir, f_name)
                with open(f_path, "rb") as f:
                    data_bytes = f.read()
                os.remove(f_path)

                self.assertFalse(weights_exist(tmpdir))
                self.assertEqual(missing_weight_files(tmpdir), [f_name])

                # Restore
                with open(f_path, "wb") as f:
                    f.write(data_bytes)
                self.assertTrue(weights_exist(tmpdir))

            # Overwrite one file with 0 bytes
            zero_target = os.path.join(tmpdir, "baseline_weights.txt")
            with open(zero_target, "w") as f:
                f.write("")
            self.assertFalse(weights_exist(tmpdir))
            self.assertIn("baseline_weights.txt", missing_weight_files(tmpdir))

    def test_10_sync_baseline_files(self):
        """10. sync_baseline_files: tự động sinh/đồng bộ .npz và .txt từ JSON."""
        sensor_cols = [f"feat_{i+1}" for i in range(5)]
        baseline_w = fit_baseline(self.X_train, sensor_cols=sensor_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Thư mục không có baseline_weights.json -> []
            synced_empty = sync_baseline_files(tmpdir)
            self.assertEqual(synced_empty, [])

            # 2. Ghi file JSON
            json_p = save_baseline(baseline_w, tmpdir)
            json_bytes_before = os.path.getsize(json_p)

            # Xóa .npz và .txt
            os.remove(os.path.join(tmpdir, "baseline_weights.npz"))
            os.remove(os.path.join(tmpdir, "baseline_weights.txt"))

            synced = sync_baseline_files(tmpdir)
            self.assertEqual(set(synced), {"baseline_weights.npz", "baseline_weights.txt"})
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "baseline_weights.npz")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "baseline_weights.txt")))
            self.assertEqual(os.path.getsize(json_p), json_bytes_before)

            # 3. Ghi sai giá trị vào npz
            corrupt_w = dict(baseline_w)
            corrupt_w["train_mean"] = [m + 1.0 for m in baseline_w["train_mean"]]
            save_baseline(corrupt_w, tmpdir)
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump(baseline_w, f, indent=2)

            synced_corrupt = sync_baseline_files(tmpdir)
            self.assertIn("baseline_weights.npz", synced_corrupt)

            # 4. Sync lại lần nữa -> []
            synced_no_op = sync_baseline_files(tmpdir)
            self.assertEqual(synced_no_op, [])



if __name__ == "__main__":
    unittest.main(verbosity=2)


"""
UNIT TESTS FOR WEIGHT SERIALIZATION & IO (ZERO SKLEARN)
======================================================
1. Round-trip model qua JSON: anomaly_score và predict giống hệt (np.allclose).
2. Round-trip baseline: baseline_score giống hệt.
3. dist_min / dist_max của baseline không đổi khi thay X_test.
4. is_cache_valid = False khi đổi random_state, đổi nội dung dữ liệu, hoặc truyền tham số CLI tường minh.
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
    save_best_model,
    load_best_model,
    weights_exist,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

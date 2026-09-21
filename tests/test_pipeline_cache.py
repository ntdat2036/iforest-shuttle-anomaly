"""
END-TO-END CACHE & WEIGHT RETRIEVAL TESTS FOR RUN_PIPELINE (ZERO SKLEARN)
========================================================================
1. Chạy lần đầu: huấn luyện và sinh đủ 4 file trọng số.
2. Chạy lần hai: LOAD trọng số mà không huấn luyện lại, không ghi đè file.
3. Thiếu file phụ (.npz/.txt): tái tạo lại từ JSON mà không train lại, hash JSON giữ nguyên.
4. Cờ --retrain: ép huấn luyện lại và ghi đè trọng số mới.
5. Thay đổi random_state: vô hiệu hóa cache và huấn luyện lại.
6. Thay đổi nội dung dữ liệu (SHA-256): vô hiệu hóa cache và huấn luyện lại.
7. Truyền tham số CLI tường minh: bỏ qua cache và huấn luyện lại.
"""

import os
import sys
import io
import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import numpy as np
import pandas as pd

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import IsolationForest
from weights_io import weights_exist, load_best_model
from run_pipeline import main


class TestPipelineCache(unittest.TestCase):

    def setUp(self):
        # 1. Lưu CWD và đăng ký cleanup os.chdir
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)

        # 2. Tạo thư mục tạm thời cho dữ liệu và trọng số
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        self.data_path = os.path.abspath(os.path.join(self.tmpdir.name, "dummy_shuttle.csv"))
        self.weights_dir = os.path.abspath(os.path.join(self.tmpdir.name, "weights"))

        # 3. Sinh dữ liệu giả 600 dòng x 10 cột KHÔNG header
        rng_x = np.random.RandomState(42)
        X_dummy = rng_x.randn(600, 9)

        rng_y = np.random.RandomState(0)
        y_dummy = rng_y.choice([1, 2, 3], size=(600, 1), p=[0.8, 0.1, 0.1])

        df_dummy = pd.DataFrame(np.hstack([X_dummy, y_dummy]))
        df_dummy.to_csv(self.data_path, header=False, index=False)

    def run_main(self, *extra_args):
        """Helper gọi hàm main() với sys.argv và stdout được capture."""
        args = [
            "run_pipeline.py",
            "--data_path", self.data_path,
            "--weights_dir", self.weights_dir,
            *extra_args
        ]
        out_buf = io.StringIO()
        with patch.object(sys, "argv", args):
            with contextlib.redirect_stdout(out_buf):
                main()
        return out_buf.getvalue()

    def snapshot(self):
        """Helper lấy snapshot {filename: (sha256_content, mtime_ns)} của 4 file trọng số."""
        target_files = [
            "baseline_weights.json",
            "baseline_weights.npz",
            "baseline_weights.txt",
            "best_model_weights.json"
        ]
        snap = {}
        for fname in target_files:
            fpath = os.path.join(self.weights_dir, fname)
            if os.path.exists(fpath):
                stat = os.stat(fpath)
                with open(fpath, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()
                snap[fname] = (content_hash, stat.st_mtime_ns)
            else:
                snap[fname] = None
        return snap

    def test_01_first_run_trains_and_writes_four_files(self):
        """1. Lần chạy đầu tiên: huấn luyện và ghi đủ 4 file trọng số vào weights_dir."""
        stdout = self.run_main("--n_estimators", "10", "--max_samples", "64")

        self.assertIn("Đã lưu trọng số", stdout)
        self.assertTrue(weights_exist(self.weights_dir))

        for fname in ["baseline_weights.json", "baseline_weights.npz", "baseline_weights.txt", "best_model_weights.json"]:
            fpath = os.path.join(self.weights_dir, fname)
            self.assertTrue(os.path.exists(fpath) and os.path.getsize(fpath) > 0)

    def test_02_second_run_loads_without_training(self):
        """2. Lần chạy thứ hai: LOAD trọng số không train lại và không ghi đè file."""
        # 1. Chạy lần đầu
        self.run_main("--n_estimators", "10", "--max_samples", "64")
        snap_before = self.snapshot()

        # 2. Chạy lần hai với patch fit để chứng minh không huấn luyện lại
        with patch.object(IsolationForest, "fit", side_effect=AssertionError("Không được train lại!")):
            stdout = self.run_main()

        self.assertIn("LOAD, bỏ qua huấn luyện", stdout)
        self.assertIn("ĐỐI CHIẾU METRICS", stdout)
        self.assertNotIn("Đã lưu trọng số", stdout)

        snap_after = self.snapshot()
        self.assertEqual(snap_before, snap_after)

    def test_03_missing_companion_files_regenerated_without_retrain(self):
        """3. Thiếu file .npz/.txt: tái tạo lại từ JSON mà không train lại và không đổi JSON."""
        # 1. Chạy lần đầu
        self.run_main("--n_estimators", "10", "--max_samples", "64")

        # 2. Xóa baseline_weights.npz và baseline_weights.txt
        npz_path = os.path.join(self.weights_dir, "baseline_weights.npz")
        txt_path = os.path.join(self.weights_dir, "baseline_weights.txt")
        os.remove(npz_path)
        os.remove(txt_path)

        with open(os.path.join(self.weights_dir, "baseline_weights.json"), "rb") as f:
            hash_json_base = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(self.weights_dir, "best_model_weights.json"), "rb") as f:
            hash_json_model = hashlib.sha256(f.read()).hexdigest()

        # 3. Chạy lần hai trong patch fit
        with patch.object(IsolationForest, "fit", side_effect=AssertionError("Không được train lại!")):
            stdout = self.run_main()

        self.assertIn("LOAD, bỏ qua huấn luyện", stdout)
        self.assertTrue(os.path.exists(npz_path) and os.path.getsize(npz_path) > 0)
        self.assertTrue(os.path.exists(txt_path) and os.path.getsize(txt_path) > 0)

        with open(os.path.join(self.weights_dir, "baseline_weights.json"), "rb") as f:
            hash_json_base_after = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(self.weights_dir, "best_model_weights.json"), "rb") as f:
            hash_json_model_after = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(hash_json_base, hash_json_base_after)
        self.assertEqual(hash_json_model, hash_json_model_after)

    def test_04_retrain_flag_overwrites(self):
        """4. Cờ --retrain: ép huấn luyện lại và ghi đè trọng số mới."""
        self.run_main("--n_estimators", "10", "--max_samples", "64")
        _, data_orig = load_best_model(self.weights_dir)
        created_at_orig = data_orig.get("meta", {}).get("created_at")

        # Chạy với --retrain
        stdout = self.run_main("--retrain", "--n_estimators", "10", "--max_samples", "64")

        self.assertIn("Đã lưu trọng số", stdout)
        _, data_new = load_best_model(self.weights_dir)
        created_at_new = data_new.get("meta", {}).get("created_at")

        self.assertNotEqual(created_at_orig, created_at_new)

    def test_05_changed_random_state_invalidates_cache(self):
        """5. Thay đổi random_state: vô hiệu hóa cache và huấn luyện lại."""
        self.run_main("--n_estimators", "10", "--max_samples", "64")

        stdout = self.run_main("--random_state", "7")
        self.assertIn("random_state không khớp", stdout)
        self.assertIn("Đã lưu trọng số", stdout)

    def test_06_changed_data_invalidates_cache(self):
        """6. Thay đổi dữ liệu giả (SHA-256): vô hiệu hóa cache và huấn luyện lại."""
        self.run_main("--n_estimators", "10", "--max_samples", "64")

        # Ghi thêm 1 dòng vào CSV giả
        with open(self.data_path, "a") as f:
            f.write("1,2,3,4,5,6,7,8,9,1\n")

        stdout = self.run_main()
        self.assertIn("SHA-256 dữ liệu không khớp", stdout)
        self.assertIn("Đã lưu trọng số", stdout)

    def test_07_explicit_cli_params_bypass_cache(self):
        """7. Truyền tham số CLI tường minh: bỏ qua cache và huấn luyện lại."""
        self.run_main("--n_estimators", "10", "--max_samples", "64")

        stdout = self.run_main("--n_estimators", "10")
        self.assertIn("Bỏ qua cache", stdout)
        self.assertIn("Đã lưu trọng số", stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

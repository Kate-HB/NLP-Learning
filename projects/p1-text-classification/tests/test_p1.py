import sys
import unittest
from pathlib import Path

import numpy as np


# 项目目录名包含连字符，测试时把目录加入模块搜索路径。
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from evaluate import calculate_metrics  # noqa: E402
from predict import build_prediction  # noqa: E402
from prepare_data import stratified_split, validate_records  # noqa: E402


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"text": f"类别{label}样本{i}", "label": label}
            for label in ("科技", "体育", "财经")
            for i in range(5)
        ]

    def test_stratified_split_preserves_every_label(self):
        splits = stratified_split(self.records, train_ratio=0.6, eval_ratio=0.2, seed=7)

        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 9,
            "eval": 3,
            "test": 3,
        })
        for rows in splits.values():
            self.assertEqual({row["label"] for row in rows}, {"科技", "体育", "财经"})

    def test_validate_records_rejects_blank_text(self):
        with self.assertRaisesRegex(ValueError, "文本不能为空"):
            validate_records([{"text": "  ", "label": "科技"}])


class EvaluationTests(unittest.TestCase):
    def test_calculate_metrics_returns_accuracy_and_macro_f1(self):
        metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f1_macro"], (2 / 3 + 0.8) / 2)

    def test_build_prediction_applies_softmax_and_label_mapping(self):
        result = build_prediction(np.array([1.0, 3.0]), {0: "体育", 1: "科技"})

        self.assertEqual(result["label"], "科技")
        self.assertAlmostEqual(result["confidence"], 0.880797, places=5)
        self.assertEqual(set(result["probabilities"]), {"体育", "科技"})


if __name__ == "__main__":
    unittest.main()

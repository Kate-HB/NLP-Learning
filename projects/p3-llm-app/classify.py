"""把 P1 的模型类适配为 P3 可调用的分类服务。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
P1_DIR = ROOT_DIR / "projects" / "p1-text-classification"
DEFAULT_MODEL_DIR = P1_DIR / "outputs" / "bert-best"


class ClassificationService:
    """懒加载 BERT；启动网页时不立刻占用显存。"""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, classifier=None):
        self.model_dir = Path(model_dir)
        self._classifier = classifier

    def _load(self):
        if self._classifier is None:
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"未找到分类模型：{self.model_dir}。请先在 P1 运行 train.py。"
                )
            if str(P1_DIR) not in sys.path:
                sys.path.insert(0, str(P1_DIR))
            text_classifier = importlib.import_module("predict").TextClassifier
            self._classifier = text_classifier(self.model_dir)
        return self._classifier

    def classify(self, text: str) -> dict:
        """稳定接口：文本输入，标签/置信度/概率输出。"""
        if not text.strip():
            raise ValueError("请输入需要分类的中文文本")
        return self._load().predict(text)

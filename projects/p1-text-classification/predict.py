"""加载已微调模型，完成单句中文文本分类。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def build_prediction(logits: np.ndarray, id2label: Mapping[int, str]) -> dict:
    """用 softmax 把无界 logits 转成总和为 1 的类别概率。"""
    logits = np.asarray(logits, dtype=float).reshape(-1)
    shifted = logits - np.max(logits)  # 减最大值可避免 exp 数值溢出。
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    predicted_id = int(np.argmax(probabilities))
    normalized_mapping = {int(index): label for index, label in id2label.items()}
    return {
        "label": normalized_mapping[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            normalized_mapping[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


class TextClassifier:
    """封装 tokenizer + model，使 UI 只关心 predict(text) 接口。"""

    def __init__(self, model_dir: str | Path, max_length: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("请输入需要分类的文本")
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0].cpu().numpy()
        return build_prediction(logits, self.model.config.id2label)


def main() -> None:
    parser = argparse.ArgumentParser(description="中文文本分类单句推理")
    parser.add_argument("text", help="待分类的中文文本")
    parser.add_argument("--model-dir", default="outputs/bert-best")
    args = parser.parse_args()
    result = TextClassifier(args.model_dir).predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

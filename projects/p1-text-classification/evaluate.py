"""计算分类指标，并对 BERT 模型执行测试集评估。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def calculate_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    """不依赖 sklearn 计算 accuracy 和 macro-F1，便于复习公式。"""
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    if labels_array.size == 0 or labels_array.shape != predictions_array.shape:
        raise ValueError("labels 和 predictions 必须长度相同且不能为空")

    accuracy = float(np.mean(labels_array == predictions_array))
    f1_scores: list[float] = []
    for class_id in sorted(set(labels_array.tolist()) | set(predictions_array.tolist())):
        true_positive = int(np.sum((labels_array == class_id) & (predictions_array == class_id)))
        false_positive = int(np.sum((labels_array != class_id) & (predictions_array == class_id)))
        false_negative = int(np.sum((labels_array == class_id) & (predictions_array != class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {"accuracy": accuracy, "f1_macro": float(np.mean(f1_scores))}


def trainer_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """把 Trainer 的 logits 转为类别编号，再复用统一指标函数。"""
    logits, labels = eval_prediction
    return calculate_metrics(labels, np.argmax(logits, axis=-1))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def evaluate_checkpoint(model_dir: Path, test_file: Path, output_dir: Path) -> dict[str, float]:
    """批量推理测试集，并保存指标与错误样本供分析。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = load_rows(test_file)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = [row["text"] for row in rows]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.cpu().numpy()

    label2id = {str(label): int(index) for label, index in model.config.label2id.items()}
    true_ids = [label2id[row["label"]] for row in rows]
    predicted_ids = np.argmax(logits, axis=-1).tolist()
    metrics = calculate_metrics(true_ids, predicted_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "errors.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "true_label", "predicted_label"])
        writer.writeheader()
        for row, true_id, predicted_id in zip(rows, true_ids, predicted_ids):
            if true_id != predicted_id:
                writer.writerow({
                    "text": row["text"],
                    "true_label": model.config.id2label[true_id],
                    "predicted_label": model.config.id2label[predicted_id],
                })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="评估训练后的中文 BERT 分类模型")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/bert-best"))
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.model_dir, args.test_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

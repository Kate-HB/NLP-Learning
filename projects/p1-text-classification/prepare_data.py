"""校验中文分类数据，并按标签分层切分 train/eval/test。"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def validate_records(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """清洗记录，并尽早报告难以定位的数据问题。"""
    cleaned: list[dict[str, str]] = []
    for line_number, record in enumerate(records, start=2):
        text = str(record.get("text", "")).strip()
        label = str(record.get("label", "")).strip()
        if not text:
            raise ValueError(f"第 {line_number} 行文本不能为空")
        if not label:
            raise ValueError(f"第 {line_number} 行标签不能为空")
        cleaned.append({"text": text, "label": label})
    if not cleaned:
        raise ValueError("数据文件不能为空")
    return cleaned


def load_csv(path: Path) -> list[dict[str, str]]:
    """读取只包含 text、label 两列的 UTF-8 CSV。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV 必须包含 text 和 label 两列")
        return validate_records(reader)


def stratified_split(
    records: list[dict[str, str]],
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """分别切分每个标签，避免小数据集的验证集缺少某一类别。"""
    if train_ratio <= 0 or eval_ratio <= 0 or train_ratio + eval_ratio >= 1:
        raise ValueError("train_ratio、eval_ratio 必须大于 0，且两者之和小于 1")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in validate_records(records):
        groups[record["label"]].append(record)

    splits = {"train": [], "eval": [], "test": []}
    rng = random.Random(seed)
    for label, rows in sorted(groups.items()):
        if len(rows) < 3:
            raise ValueError(f"标签“{label}”至少需要 3 条样本，才能进入三个数据集")
        rows = rows.copy()
        rng.shuffle(rows)

        # 验证集和测试集至少各留一条；其余优先用于训练。
        train_count = min(max(1, round(len(rows) * train_ratio)), len(rows) - 2)
        eval_count = min(max(1, round(len(rows) * eval_ratio)), len(rows) - train_count - 1)
        splits["train"].extend(rows[:train_count])
        splits["eval"].extend(rows[train_count : train_count + eval_count])
        splits["test"].extend(rows[train_count + eval_count :])

    # 打散不同标签，避免训练时同类样本连续出现。
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    """写入标准化后的数据，供 baseline 和 BERT 共用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="分层切分中文文本分类数据")
    parser.add_argument("--input", type=Path, default=Path("data/raw/news.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = stratified_split(
        load_csv(args.input), args.train_ratio, args.eval_ratio, args.seed
    )
    for split_name, rows in splits.items():
        write_csv(args.output_dir / f"{split_name}.csv", rows)
        print(f"{split_name}: {len(rows)} 条")


if __name__ == "__main__":
    main()

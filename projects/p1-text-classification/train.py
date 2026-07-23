"""使用 Hugging Face Trainer 微调中文 BERT 分类模型。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate import calculate_metrics


def read_split(path: Path, label2id: dict[str, int]) -> list[dict]:
    """把可读标签转换为模型训练所需的整数 labels。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [
            {"text": row["text"], "labels": label2id[row["label"]]}
            for row in csv.DictReader(file)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="微调中文 BERT 文本分类模型")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert"))
    parser.add_argument("--final-dir", type=Path, default=Path("outputs/bert-best"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    with (args.data_dir / "train.csv").open("r", encoding="utf-8-sig", newline="") as file:
        labels = sorted({row["label"] for row in csv.DictReader(file)})
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = {
        name: Dataset.from_list(read_split(args.data_dir / f"{name}.csv", label2id))
        for name in ("train", "eval")
    }

    def tokenize(batch: dict) -> dict:
        # 动态 padding 在 DataCollator 中完成，这里只截断超长文本。
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = {name: dataset.map(tokenize, batched=True) for name, dataset in datasets.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
    )

    def compute_metrics(prediction) -> dict[str, float]:
        predictions = np.argmax(prediction.predictions, axis=-1)
        return calculate_metrics(prediction.label_ids, predictions)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.final_dir)
    tokenizer.save_pretrained(args.final_dir)
    args.final_dir.mkdir(parents=True, exist_ok=True)
    (args.final_dir / "label_map.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

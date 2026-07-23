"""训练字符 TF-IDF baseline，验证数据和评估流程是否合理。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate import calculate_metrics


def read_rows(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return [row["text"] for row in rows], [row["label"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="字符 TF-IDF 文本分类 baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train_texts, train_labels = read_rows(args.data_dir / "train.csv")
    test_texts, test_labels = read_rows(args.data_dir / "test.csv")
    pipeline = Pipeline([
        # 中文不以空格分词，字符 1~2 gram 是简单且稳定的 baseline。
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    classes = sorted(set(train_labels))
    label2id = {label: index for index, label in enumerate(classes)}
    metrics = calculate_metrics(
        [label2id[label] for label in test_labels],
        [label2id[label] for label in predictions],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output_dir / "model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

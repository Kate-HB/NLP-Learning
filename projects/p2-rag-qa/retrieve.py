"""加载 NumPy 向量索引，并执行余弦相似度 Top-k 检索。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingest import Embedder, SentenceTransformerEmbedder


@dataclass(frozen=True)
class VectorIndex:
    vectors: np.ndarray
    metadata: list[dict]


def load_index(index_dir: Path) -> VectorIndex:
    """同时加载向量与元数据，并检查两者是否仍一一对应。"""
    vectors = np.load(index_dir / "vectors.npy")
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    if vectors.ndim != 2 or len(metadata) != vectors.shape[0]:
        raise ValueError("索引损坏：向量数量与元数据数量不一致")
    return VectorIndex(vectors=vectors.astype(np.float32), metadata=metadata)


def cosine_scores(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """余弦相似度只比较方向，可减弱文本长度导致的向量模长差异。"""
    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("query 向量维度与索引维度不一致")
    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denominator = vector_norms * query_norm
    return np.divide(
        vectors @ query_vector,
        denominator,
        out=np.zeros(vectors.shape[0], dtype=np.float32),
        where=denominator > 0,
    )


def search(query: str, index: VectorIndex, embedder: Embedder, top_k: int = 3) -> list[dict]:
    """返回按相关度降序排列的 chunk、来源和分数。"""
    if not query.strip():
        raise ValueError("查询问题不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    query_vector = np.asarray(embedder.encode([query]))[0]
    scores = cosine_scores(query_vector, index.vectors)
    ranked_ids = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
    return [
        {**index.metadata[int(index_id)], "score": float(scores[index_id])}
        for index_id in ranked_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="检索与问题最相关的本地文档片段")
    parser.add_argument("query")
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    parser.add_argument("--model-name", default="shibing624/text2vec-base-chinese")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    results = search(
        args.query,
        load_index(args.index_dir),
        SentenceTransformerEmbedder(args.model_name),
        args.top_k,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

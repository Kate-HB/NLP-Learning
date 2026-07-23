"""读取本地文档、切分 chunk、生成向量并持久化索引。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """只规定 RAG 所需的最小向量模型接口，方便替换和测试。"""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """使用 SentenceTransformer 生成能表达整句语义的稠密向量。"""

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """用重叠字符窗口切块；重叠可避免答案恰好落在边界时丢失上下文。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 0，且 overlap 满足 0 <= overlap < chunk_size")
    # 保留段落边界，但清除空行两侧多余空白。
    normalized = "\n\n".join(part.strip() for part in text.split("\n\n") if part.strip())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """递归读取 Markdown/TXT，并用相对路径作为可追踪来源。"""
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            documents.append((path.relative_to(data_dir).as_posix(), path.read_text(encoding="utf-8")))
    if not documents:
        raise ValueError(f"{data_dir} 中没有 .md 或 .txt 文档")
    return documents


def build_index(
    data_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    chunk_size: int = 300,
    overlap: int = 50,
) -> int:
    """把向量和文本元数据分开保存，便于观察索引内部结构。"""
    metadata: list[dict] = []
    for source, text in load_documents(data_dir):
        for chunk_id, chunk in enumerate(split_text(text, chunk_size, overlap)):
            metadata.append({"source": source, "chunk_id": chunk_id, "text": chunk})
    if not metadata:
        raise ValueError("文档切分后没有可索引内容")

    vectors = np.asarray(embedder.encode([item["text"] for item in metadata]), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(metadata):
        raise ValueError("embedding 数量必须与 chunk 数量一致")

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", vectors)
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (index_dir / "manifest.json").write_text(
        json.dumps({"chunks": len(metadata), "dimension": vectors.shape[1]}, indent=2),
        encoding="utf-8",
    )
    return len(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="建立本地 RAG 向量索引")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    parser.add_argument("--model-name", default="shibing624/text2vec-base-chinese")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    count = build_index(
        args.data_dir,
        args.index_dir,
        SentenceTransformerEmbedder(args.model_name),
        args.chunk_size,
        args.overlap,
    )
    print(f"索引完成：{count} 个 chunks，保存到 {args.index_dir}")


if __name__ == "__main__":
    main()

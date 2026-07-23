"""把 P2 的索引、检索和生成链适配为 P3 问答服务。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
P2_DIR = ROOT_DIR / "projects" / "p2-rag-qa"
DEFAULT_INDEX_DIR = P2_DIR / "outputs" / "index"


class _P2Backend:
    """集中保存 P2 的模型与索引，避免每次点击都重新加载。"""

    def __init__(self, index_dir: Path, model_name: str, min_score: float):
        if not index_dir.exists():
            raise FileNotFoundError(f"未找到 RAG 索引：{index_dir}。请先在 P2 运行 ingest.py。")
        if str(P2_DIR) not in sys.path:
            sys.path.insert(0, str(P2_DIR))
        ingest = importlib.import_module("ingest")
        retrieve = importlib.import_module("retrieve")
        ask = importlib.import_module("ask")
        self.index = retrieve.load_index(index_dir)
        self.embedder = ingest.SentenceTransformerEmbedder(model_name)
        self.generator = ask.generator_from_environment()
        self.answer_question = ask.answer_question
        self.min_score = min_score

    def answer(self, question: str, top_k: int) -> dict:
        return self.answer_question(
            question,
            self.index,
            self.embedder,
            self.generator,
            top_k=top_k,
            min_score=self.min_score,
        )


class RAGService:
    """懒加载索引和 embedding 模型，并暴露统一 answer 接口。"""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        model_name: str = "shibing624/text2vec-base-chinese",
        min_score: float = 0.15,
        backend=None,
    ):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.min_score = min_score
        self._backend = backend

    def _load(self):
        if self._backend is None:
            self._backend = _P2Backend(self.index_dir, self.model_name, self.min_score)
        return self._backend

    def answer(self, question: str, top_k: int = 3) -> dict:
        if not question.strip():
            raise ValueError("请输入需要查询的问题")
        return self._load().answer(question, int(top_k))

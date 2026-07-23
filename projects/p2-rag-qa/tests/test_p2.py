import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from ask import answer_question, build_prompt  # noqa: E402
from ingest import build_index, split_text  # noqa: E402
from retrieve import load_index, search  # noqa: E402


class FakeEmbedder:
    """测试向量是人为可解释的，不依赖联网下载真实模型。"""

    def encode(self, texts):
        vectors = []
        for text in texts:
            vectors.append([
                float("BERT" in text or "编码器" in text),
                float("RAG" in text or "检索" in text),
            ])
        return np.asarray(vectors, dtype=np.float32)


class IngestTests(unittest.TestCase):
    def test_split_text_keeps_overlap_and_drops_blank_chunks(self):
        chunks = split_text("abcdefghijklmnop", chunk_size=10, overlap=3)

        self.assertEqual(chunks, ["abcdefghij", "hijklmnop"])
        self.assertEqual(chunks[0][-3:], chunks[1][:3])

    def test_build_and_load_index_preserve_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs = root / "docs"
            docs.mkdir()
            (docs / "bert.md").write_text("BERT 是双向编码器。", encoding="utf-8")

            count = build_index(docs, root / "index", FakeEmbedder(), chunk_size=50, overlap=5)
            index = load_index(root / "index")

            self.assertEqual(count, 1)
            self.assertEqual(index.metadata[0]["source"], "bert.md")
            self.assertEqual(index.vectors.shape, (1, 2))


class RetrieveAndAnswerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        docs = root / "docs"
        docs.mkdir()
        (docs / "knowledge.md").write_text(
            "BERT 是双向编码器。\n\nRAG 先检索文档，再生成答案。", encoding="utf-8"
        )
        build_index(docs, root / "index", FakeEmbedder(), chunk_size=12, overlap=2)
        self.index = load_index(root / "index")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_returns_highest_cosine_similarity_first(self):
        results = search("RAG 如何检索", self.index, FakeEmbedder(), top_k=2)

        self.assertIn("RAG", results[0]["text"])
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_answer_rejects_when_retrieval_score_is_too_low(self):
        result = answer_question(
            "天气如何", self.index, FakeEmbedder(), generator=lambda _: "不应调用", min_score=0.2
        )

        self.assertIn("未找到足够依据", result["answer"])
        self.assertEqual(result["citations"], [])

    def test_answer_contains_numbered_citations(self):
        result = answer_question(
            "RAG 如何检索", self.index, FakeEmbedder(), generator=lambda _: "先检索再生成 [1]。"
        )

        self.assertEqual(result["answer"], "先检索再生成 [1]。")
        self.assertEqual(result["citations"][0]["id"], 1)
        self.assertIn("[1]", build_prompt("什么是 RAG", result["retrieved"]))


if __name__ == "__main__":
    unittest.main()

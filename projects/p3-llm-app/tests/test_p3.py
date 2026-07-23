import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import classification_view, rag_view  # noqa: E402
from classify import ClassificationService  # noqa: E402
from rag import RAGService  # noqa: E402


class FakeClassifier:
    def predict(self, text):
        return {
            "label": "科技",
            "confidence": 0.8,
            "probabilities": {"科技": 0.8, "财经": 0.2},
        }


class FakeRAGBackend:
    def answer(self, question, top_k):
        return {
            "answer": "RAG 先检索再生成 [1]。",
            "citations": [{"id": 1, "source": "rag.md", "chunk_id": 0, "score": 0.91}],
            "retrieved": [{"text": "RAG 先检索。", "score": 0.91}],
        }


class ServiceTests(unittest.TestCase):
    def test_classification_service_accepts_injected_backend(self):
        result = ClassificationService(classifier=FakeClassifier()).classify("发布新芯片")

        self.assertEqual(result["label"], "科技")

    def test_rag_service_accepts_injected_backend(self):
        result = RAGService(backend=FakeRAGBackend()).answer("什么是 RAG", top_k=2)

        self.assertEqual(result["citations"][0]["source"], "rag.md")


class ViewTests(unittest.TestCase):
    def test_classification_view_displays_uncertainty(self):
        summary, probabilities = classification_view(FakeClassifier().predict("任意文本"))

        self.assertIn("80.00%", summary)
        self.assertEqual(probabilities["科技"], 0.8)

    def test_rag_view_displays_source_and_score(self):
        answer, citations = rag_view(FakeRAGBackend().answer("问题", 3))

        self.assertIn("[1]", answer)
        self.assertIn("rag.md#chunk-0", citations)
        self.assertIn("0.9100", citations)


if __name__ == "__main__":
    unittest.main()

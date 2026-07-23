"""组合检索结果与生成器，输出有依据、带引用的回答。"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Callable

from ingest import Embedder, SentenceTransformerEmbedder
from retrieve import VectorIndex, load_index, search


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """给每个上下文编号，要求生成内容用同样编号标注证据。"""
    context = "\n\n".join(
        f"[{number}] 来源：{item['source']}#chunk-{item['chunk_id']}\n{item['text']}"
        for number, item in enumerate(retrieved, start=1)
    )
    return (
        "你是本地知识库问答助手。只能根据给定上下文回答；证据不足时明确说不知道。"
        "每个事实后使用 [数字] 标注来源，不要编造引用。\n\n"
        f"上下文：\n{context}\n\n问题：{question}\n回答："
    )


def extractive_answer(retrieved: list[dict]) -> str:
    """无外部 LLM 时返回最相关原文，保证项目离线可运行且不会幻觉。"""
    return "根据最相关文档：\n\n" + "\n\n".join(
        f"{item['text']} [{number}]" for number, item in enumerate(retrieved, start=1)
    )


class OpenAICompatibleGenerator:
    """通过标准 HTTP 调用 OpenAI 兼容接口，不把密钥写进源码。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"].strip()


def answer_question(
    question: str,
    index: VectorIndex,
    embedder: Embedder,
    generator: Callable[[str], str] | None = None,
    top_k: int = 3,
    min_score: float = 0.15,
) -> dict:
    """先检索再生成；低于阈值直接拒答，防止无依据生成。"""
    retrieved = search(question, index, embedder, top_k)
    if not retrieved or retrieved[0]["score"] < min_score:
        return {
            "answer": "未找到足够依据，请换一种问法或补充知识库文档。",
            "citations": [],
            "retrieved": retrieved,
        }

    answer = generator(build_prompt(question, retrieved)) if generator else extractive_answer(retrieved)
    citations = [
        {
            "id": number,
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "score": item["score"],
        }
        for number, item in enumerate(retrieved, start=1)
    ]
    return {"answer": answer, "citations": citations, "retrieved": retrieved}


def generator_from_environment() -> OpenAICompatibleGenerator | None:
    """三个变量齐全才启用远程生成，否则保持离线摘录模式。"""
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not all((base_url, api_key, model)):
        return None
    return OpenAICompatibleGenerator(base_url, api_key, model)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地文档 RAG 问答")
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    parser.add_argument("--model-name", default="shibing624/text2vec-base-chinese")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.15)
    args = parser.parse_args()
    result = answer_question(
        args.question,
        load_index(args.index_dir),
        SentenceTransformerEmbedder(args.model_name),
        generator_from_environment(),
        args.top_k,
        args.min_score,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

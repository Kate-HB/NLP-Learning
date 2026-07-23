"""P1 分类 + P2 RAG 的双 Tab Gradio 应用入口。"""

from __future__ import annotations

from classify import ClassificationService
from rag import RAGService


def classification_view(result: dict) -> tuple[str, dict[str, float]]:
    """显式显示置信度；概率接近时提醒用户模型不确定。"""
    confidence = float(result["confidence"])
    warning = "\n\n> 置信度较低，请人工复核。" if confidence < 0.6 else ""
    summary = f"### 预测：{result['label']}\n\n置信度：{confidence:.2%}{warning}"
    return summary, result["probabilities"]


def rag_view(result: dict) -> tuple[str, str]:
    """把引用来源和分数放在答案旁边，方便核验而非盲信。"""
    if not result["citations"]:
        return result["answer"], "无引用：检索结果未达到阈值。"
    citations = "\n".join(
        f"- [{item['id']}] `{item['source']}#chunk-{item['chunk_id']}`，相似度 {item['score']:.4f}"
        for item in result["citations"]
    )
    return result["answer"], citations


def create_app(
    classification_service: ClassificationService | None = None,
    rag_service: RAGService | None = None,
):
    """创建界面但不启动服务器，便于测试或被其他程序挂载。"""
    import gradio as gr

    classification_service = classification_service or ClassificationService()
    rag_service = rag_service or RAGService()

    def classify_handler(text: str):
        try:
            return classification_view(classification_service.classify(text))
        except Exception as error:
            return f"### 错误\n\n{error}", {}

    def rag_handler(question: str, top_k: int):
        try:
            return rag_view(rag_service.answer(question, top_k))
        except Exception as error:
            return f"错误：{error}", ""

    with gr.Blocks(title="NLP 学习项目 Demo") as demo:
        gr.Markdown("# NLP 学习项目 Demo\n分类结果和 RAG 回答均展示不确定性或证据来源。")
        with gr.Tab("中文文本分类"):
            text = gr.Textbox(label="中文文本", placeholder="例如：国产芯片公司发布新处理器")
            classify_button = gr.Button("开始分类", variant="primary")
            classification_summary = gr.Markdown()
            probabilities = gr.Label(label="各类别概率", num_top_classes=10)
            classify_button.click(
                classify_handler,
                inputs=text,
                outputs=[classification_summary, probabilities],
            )
        with gr.Tab("RAG 文档问答"):
            question = gr.Textbox(label="问题", placeholder="例如：RAG 为什么能降低幻觉？")
            top_k = gr.Slider(1, 8, value=3, step=1, label="Top-k")
            ask_button = gr.Button("检索并回答", variant="primary")
            answer = gr.Markdown(label="回答")
            citations = gr.Markdown(label="引用来源")
            ask_button.click(rag_handler, inputs=[question, top_k], outputs=[answer, citations])
    return demo


if __name__ == "__main__":
    create_app().launch()

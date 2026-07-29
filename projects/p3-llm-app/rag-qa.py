from transformers import pipeline

model = pipeline(model="Kate-lf/rag-qa-base-bert")


def predict(context, question):
    result = model(question=question, context=context)
    score = result["score"]
    answer = result["answer"]
    if score < 0.1 or not answer.strip():
        return "上下文不包含答案，无法回答。"
    return f"答案：{answer}\n\n置信度：{score:.2%}"


import gradio as gr

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="上下文", lines=5, placeholder="粘贴一段文本作为上下文..."),
        gr.Textbox(label="问题", placeholder="根据上下文提问..."),
    ],
    outputs="text",
    title="抽取式问答",
    description="输入上下文和问题，模型从上下文中抽取答案。上下文不包含答案时返回'无法回答'。",
    examples=[
        ["苏轼是北宋著名文学家、书画家，号东坡居士，眉州眉山人。", "苏轼的号是什么？"],
        ["Python由Guido van Rossum于1991年首次发布。", "Python是什么时候发布的？"],
    ],
)

if __name__ == "__main__":
    demo.launch()

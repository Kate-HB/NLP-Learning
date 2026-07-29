from transformers import pipeline

model = pipeline("text-classification", model="Kate-lf/emotion-classification")


def predict(text):
    result = model(text)[0]
    return f"{result['label']} (置信度: {result['score']:.2%})"


import gradio as gr

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title="句子情绪分类",
    description="输入句子，判断属于 伤心 / 关心 / 厌恶 / 平静 / 惊讶 / 开心 / 生气 / 疑问",
    examples=[["曾经最好的朋友，现在连点赞都不会了。"], ["你在我最无助的时候给了我依靠！"]],
).launch()

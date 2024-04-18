from openai import OpenAI
import gradio as gr

client = OpenAI(
    api_key="EMPTY",
    base_url="http://ali-ai:8000/v1",
)


def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="Mixtral-8x7B-Instruct-v0.1",
        messages=history_openai_format,
        temperature=1.0,
        stream=True,
    )

    partial_message = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            partial_message = partial_message + content
            yield partial_message


gr.ChatInterface(predict).launch()

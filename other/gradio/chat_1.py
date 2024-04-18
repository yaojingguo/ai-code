import gradio as gr

def echo(message, history):
    return f"your message is {message}"

demo = gr.ChatInterface(fn=echo, examples=["hello", "hola", "merhaba"], title="Echo Bot")
demo.launch()
from transformers import pipeline

model_id = "yaojingguo/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)


def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs

import gradio as gr

demo = gr.Interface(
    fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.Label()
)
demo.launch(debug=True, server_name="172.20.197.2")

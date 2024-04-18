## litelvm
vllm server:
```bash
$ cd /data/yaojg/code/llm/models
$ python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 2 \
    --load-format safetensors
```

```bash
nohup python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 2 \
    --load-format safetensors \
    > vllm.log 2>&1 </dev/null &
```


SDK: `python lite.py`

Run proxy with `litellm --config config.yaml`. Use the proxy:
```bash
curl --location 'http://localhost:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "vllm-model",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```

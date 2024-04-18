import openai
client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="vllm-model", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)



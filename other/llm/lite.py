import litellm

messages = [
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    {"role": "user", "content": "Where was it played?"},
]


response = litellm.completion(
    api_key = "EMPTY",
    model="Mixtral-8x7B-Instruct-v0.1",  # add a vllm prefix so litellm knows the custom_llm_provider==vllm
    messages=messages,
    custom_llm_provider="openai",
    api_base="http://ali-ai:8000/v1",
    temperature=0.2,
    max_tokens=80,
)

print(response.choices[0].message.content)

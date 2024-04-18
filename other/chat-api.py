import openai
import time

openai.api_base = "http://172.20.193.39/v1"
openai.api_key = "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"

model = "gpt-3.5-turbo-0301"

# response = openai.ChatCompletion.create(
#     model=model,
#     messages=example_messages,
#     temperature=0,
#     max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
# )

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        n = 2,
        temperature = 0, # this is the degree of randomness of the model's output
    )
    print(f"choices count: {len(response.choices)}")
    for entry in response.choices:
        print(f"-- {entry}")
    return response.choices[0].message["content"]

# Example of an OpenAI ChatCompletion request
# https://platform.openai.com/docs/guides/chat

# record the time before the request is sent
start_time = time.time()

# send a ChatCompletion request to count to 100
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': 'Count to 10, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
    ],
    temperature=0,
    stream=True
)

for chunk in response:
    print(chunk)


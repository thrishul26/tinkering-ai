# Basic query testing (Chat Completions API) - Open AI SDK

import os
from openai import OpenAI
from dotenv import load_dotenv
from zscaler_proxy import main
load_dotenv()

http_client = main()
client = OpenAI(
    api_key = os.getenv("PERPLEXITY_API_KEY"),
    base_url = "https://api.perplexity.ai",
    http_client=http_client
)

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What are the benefits of learning Python"}
]

# Chat Completions API
response = client.chat.completions.create(
    model = "sonar",
    messages = messages,
    temperature = 0.7,
    max_tokens = 300
)

print(response.choices[0].message.content)
print(response.usage.prompt_tokens)
print(response.usage.completion_tokens)
print(response.usage.total_tokens)
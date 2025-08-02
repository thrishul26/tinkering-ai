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
    {"role": "user", "content": "What is the latest in AI research"}
]

stream = client.chat.completions.create(
    model = "sonar",
    messages = "messages",
    stream = True, # Only diff between stream and non stream api methods
    max_tokens = 300,
    temperature = 0.7,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
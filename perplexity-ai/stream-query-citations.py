# Responses API is used

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
    {"role": "user", "content": "What are some latest advancements in AI Research"}
]

stream = client.responses.create(
    model= "sonar",
    messages= messages,
    stream = True,
    temperature= 0.7,
    max_tokens = 300
)

content = ""
citations = []
usage_info = None
search_results = None
finish_reason = None

for chunk in stream:
    # Content arrives progressively
    if chunk.output_text is not None:
        content_chunk = chunk.output_text
        content+= content_chunk
        print(content_chunk, end="")

    # collect citations
    if hasattr(chunk, 'citations') and chunk.citations:
        citations = chunk.citations
    
    # collect usage
    if hasattr(chunk, 'usage') and chunk.usage:
        usage_info = chunk.usage

    if hasattr(chunk, 'search_results') and chunk.search_results:
        search_results = chunk.search_results
    
    # Handle completion
    if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
        finish_reason = chunk.finish_reason


print(f"\n\nFinish reason: {chunk.finish_reason}")
print(f"Usage: {usage_info}")
        
if search_results:
    print("\n\n Sources:")
    print("=" * 50)

    for i, source in enumerate(search_results, 1):
        print(f"\n{i}. {source['title']}")
        print(f"       {source['url']}")
        if source.get('date'):
            print(f"    Published: {source['date']}")
        if source.get('last_updated'):
            print(f"    Last Updated: {source['last_updated']}")
        print("-" * 40)
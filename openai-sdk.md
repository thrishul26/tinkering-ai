# OpenAI-Compatible Python SDK (v1+) — Universal Cheat Sheet

This guide helps you use the OpenAI Python SDK (v1+) in a provider-neutral way, so your code works across platforms like OpenAI, DeepSeek, Groq, Together AI, Fireworks, and others.

---

## 1. Installation

```bash
pip install --upgrade openai
```

---

## 2. Client Initialization

### General Pattern (Recommended)
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.example.com/v1"  # Optional for OpenAI, required for other providers
)
```

### Examples
```python
# OpenAI
client = OpenAI(api_key="sk-...")

# DeepSeek
client = OpenAI(
    api_key="ds-key",
    base_url="https://api.deepseek.com/v1"
)

# Groq
client = OpenAI(
    api_key="groq-key",
    base_url="https://api.groq.com/openai/v1"
)
```

---

## 3. Chat Completion (Synchronous)

```python
response = client.chat.completions.create(
    model="gpt-4",  # or other compatible model names
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

---

## 4. Chat Completion (Streaming)

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Write a poem about rain."}
    ],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 5. Async Support

```python
from openai import AsyncOpenAI
import asyncio

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## 6. Model Selection and Base URLs

| Model Name            | Provider     | base_url Required | Notes                           |
|----------------------|--------------|-------------------|----------------------------------|
| `gpt-4`              | OpenAI       | No                | Chat model                       |
| `gpt-3.5-turbo`      | OpenAI       | No                | Cost-efficient chat model        |
| `deepseek-chat`      | DeepSeek     | Yes               | Math/code optimized              |
| `mistral-7b-instruct`| Together AI  | Yes               | Small open-source chat model     |
| `llama3-70b`         | Groq         | Yes               | High-quality open-source model   |

> Check each provider’s documentation for latest models and endpoints.

---

## 7. Optional Parameters

| Parameter        | Purpose                                                |
|------------------|--------------------------------------------------------|
| `temperature`    | Controls randomness (0 = deterministic, 1 = creative)  |
| `top_p`          | Nucleus sampling; alternative to temperature          |
| `max_tokens`     | Max tokens in the response                            |
| `stop`           | List of stop sequences                                |
| `stream`         | If True, yields response chunks                       |
| `response_format`| Set to "json" for function/tool-style outputs         |

---

## 8. Multi-Provider Client Factory (Example)

```python
def get_client(provider: str):
    if provider == "openai":
        return OpenAI(api_key="sk-openai")
    elif provider == "deepseek":
        return OpenAI(
            api_key="ds-key",
            base_url="https://api.deepseek.com/v1"
        )
    elif provider == "groq":
        return OpenAI(
            api_key="groq-key",
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        raise ValueError("Unknown provider")

client = get_client("deepseek")
response = client.chat.completions.create(...)
```

---

## 9. Completion API (Legacy)

```python
response = client.completions.create(
    model="text-davinci-003",
    prompt="Translate this into French: Hello.",
    max_tokens=60
)
print(response.choices[0].text)
```

> Use only for legacy GPT-3 models. For all chat models, prefer `chat.completions.create`.

---

## 10. References

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [LangChain LLM Integration](https://docs.langchain.com/docs/integrations/llms/openai/)
- [DeepSeek Docs](https://platform.deepseek.com)
- [Groq Docs](https://console.groq.com)
- [Together AI Docs](https://docs.together.ai/)

---

Feel free to use this as a base for building reusable agents, multi-provider wrappers, or integration layers.
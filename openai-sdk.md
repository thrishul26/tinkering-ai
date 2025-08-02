# OpenAI-Compatible Python SDK (v1+) — Comprehensive Developer Guide

This comprehensive guide helps you master the OpenAI Python SDK (v1+) in a provider-neutral way, so your code works across platforms like OpenAI, DeepSeek, Groq, Together AI, Fireworks, and others.

---

## 1. Installation & Setup

### Basic Installation
```bash
pip install --upgrade openai
```

### Optional Dependencies for Advanced Features
```bash
# For async support and better performance
pip install --upgrade openai[async]

# For Pydantic models (structured outputs)
pip install --upgrade pydantic

# For data science workflows
pip install --upgrade pandas numpy

# For embeddings and vector operations
pip install --upgrade numpy scikit-learn
```

---

## 2. Client Initialization

### General Pattern (Recommended)
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.example.com/v1",  # Optional for OpenAI, required for other providers
    timeout=30.0,  # Request timeout in seconds
    max_retries=3  # Automatic retry on failures
)
```

### Environment Variables Setup
```python
import os
from openai import OpenAI

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-key-here"
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"  # Optional

# Client will automatically use environment variables
client = OpenAI()
```

### Provider Examples
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

# Together AI
client = OpenAI(
    api_key="together-key",
    base_url="https://api.together.xyz/v1"
)

# Azure OpenAI
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="azure-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-10-21"
)
```

---

## 3. Chat Completion (Synchronous)

### Basic Example
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### Advanced Parameters
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,          # Creativity (0.0-2.0)
    max_tokens=1000,          # Response length limit
    top_p=0.9,               # Nucleus sampling
    frequency_penalty=0.1,    # Reduce repetition (-2.0 to 2.0)
    presence_penalty=0.1,     # Encourage topic diversity (-2.0 to 2.0)
    stop=["END", "STOP"],    # Stop sequences
    seed=42,                 # For reproducible outputs
    logprobs=True,           # Include log probabilities
    top_logprobs=5           # Top alternative tokens
)

# Access detailed response information
choice = response.choices[0]
print(f"Content: {choice.message.content}")
print(f"Finish reason: {choice.finish_reason}")
print(f"Usage: {response.usage}")
```

---

## 4. Chat Completion (Streaming)

### Basic Streaming
```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Write a poem about rain."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Advanced Streaming with Error Handling
```python
def stream_chat_completion(messages, model="gpt-4"):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                
        return full_response
        
    except Exception as e:
        print(f"Streaming error: {e}")
        return None
```

---

## 5. Async Support

### Basic Async Client
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

### Concurrent Requests
```python
import asyncio
from openai import AsyncOpenAI

async def process_multiple_requests():
    client = AsyncOpenAI()
    
    questions = [
        "What is Python?",
        "Explain machine learning",
        "How does the internet work?"
    ]
    
    tasks = []
    for question in questions:
        task = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}],
            max_tokens=100
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Q: {questions[i]}")
        print(f"A: {response.choices[0].message.content}\n")

asyncio.run(process_multiple_requests())
```

### Async Streaming
```python
async def async_stream_example():
    client = AsyncOpenAI()
    
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Count to 10"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
```

---

## 6. Function Calling

### Basic Function Definition
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"  # Let model decide when to call functions
)

# Check if function was called
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### Advanced Function Calling with Implementation
```python
import json

def get_weather(location, unit="fahrenheit"):
    """Mock weather function"""
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "description": "Sunny"
    }

def run_conversation():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        available_functions = {"get_weather": get_weather}
        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response)
            })
        
        second_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return second_response.choices[0].message.content

print(run_conversation())
```

---

## 7. Structured Outputs with Pydantic

### Basic Structured Output
```python
from pydantic import BaseModel
from typing import List

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}
    ],
    response_format=CalendarEvent
)

event = completion.choices[0].message.parsed
print(f"Event: {event.name}")
print(f"Date: {event.date}")
print(f"Participants: {event.participants}")
```

### Complex Structured Output with Nested Models
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Person(BaseModel):
    name: str
    email: str
    role: Optional[str] = None

class Task(BaseModel):
    title: str = Field(description="The task title")
    description: str = Field(description="Detailed task description")
    priority: Priority
    assigned_to: List[Person]
    due_date: Optional[str] = Field(description="Due date in YYYY-MM-DD format")
    estimated_hours: Optional[int] = Field(description="Estimated hours to complete")

class ProjectPlan(BaseModel):
    project_name: str
    tasks: List[Task]
    total_estimated_hours: int

# Use the structured output
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a project manager. Create a project plan."},
        {"role": "user", "content": "Create a plan for building a mobile app with login, dashboard, and payment features. Team: Alice (developer), Bob (designer), Carol (tester)."}
    ],
    response_format=ProjectPlan
)

project = completion.choices[0].message.parsed
print(f"Project: {project.project_name}")
for task in project.tasks:
    print(f"- {task.title} ({task.priority.value}) - {len(task.assigned_to)} people")
```

### Handling Refusals
```python
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "Generate harmful content"}
    ],
    response_format=CalendarEvent
)

if completion.choices[0].message.refusal:
    print("Model refused:", completion.choices[0].message.refusal)
else:
    event = completion.choices[0].message.parsed
    print("Parsed event:", event)
```

---

## 8. Embeddings

### Basic Embeddings
```python
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

embedding = get_embedding("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Batch Embeddings
```python
def get_embeddings_batch(texts, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [data.embedding for data in response.data]

texts = [
    "Python is a programming language",
    "Machine learning is fascinating",
    "Natural language processing"
]

embeddings = get_embeddings_batch(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Similarity Search with Embeddings
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_search(query, documents, model="text-embedding-ada-002"):
    # Get embeddings for query and documents
    all_texts = [query] + documents
    embeddings = get_embeddings_batch(all_texts, model)
    
    query_embedding = np.array(embeddings[0]).reshape(1, -1)
    doc_embeddings = np.array(embeddings[1:])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Sort by similarity
    results = list(zip(documents, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

# Example usage
documents = [
    "Python is a high-level programming language",
    "Machine learning algorithms learn from data",
    "Natural language processing deals with text",
    "Computer vision analyzes images and videos"
]

query = "What is Python?"
results = semantic_search(query, documents)

for doc, similarity in results:
    print(f"Similarity: {similarity:.3f} - {doc}")
```

---

## 9. Error Handling & Best Practices

### Comprehensive Error Handling
```python
from openai import OpenAI, OpenAIError, RateLimitError, APIError
import time
import random

def robust_chat_completion(messages, max_retries=3):
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                timeout=30
            )
            return response
            
        except RateLimitError as e:
            print(f"Rate limit exceeded. Attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise e
                
        except APIError as e:
            print(f"API error: {e}")
            if e.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
                
        except OpenAIError as e:
            print(f"OpenAI error: {e}")
            raise e
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
```

### Rate Limiting and Usage Tracking
```python
import time
from datetime import datetime, timedelta
from collections import deque

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
    
    def wait_if_needed(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        while self.requests and self.requests[0] < now - timedelta(minutes=1):
            self.requests.popleft()
        
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]).seconds
            print(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
        
        self.requests.append(now)

# Usage
rate_limiter = RateLimiter(max_requests_per_minute=50)

def safe_api_call(messages):
    rate_limiter.wait_if_needed()
    return client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

### Token Usage Monitoring
```python
class UsageTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
    
    def track_usage(self, response):
        if hasattr(response, 'usage'):
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            self.total_requests += 1
    
    def get_stats(self):
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        return {
            "total_requests": self.total_requests,
            "total_tokens": total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "avg_tokens_per_request": total_tokens / max(self.total_requests, 1)
        }

# Usage
tracker = UsageTracker()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

tracker.track_usage(response)
print(tracker.get_stats())
```

---

## 10. Advanced Features

### Content Moderation
```python
def moderate_content(text):
    response = client.moderations.create(input=text)
    result = response.results[0]
    
    if result.flagged:
        print("Content flagged for:")
        for category, flagged in result.categories.model_dump().items():
            if flagged:
                print(f"- {category}")
        return False
    return True

# Example usage
user_input = "This is a normal message"
if moderate_content(user_input):
    # Proceed with API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}]
    )
```

### Image Analysis (Vision)
```python
def analyze_image(image_url, prompt="What's in this image?"):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

# Example usage
image_url = "https://example.com/image.jpg"
description = analyze_image(image_url, "Describe this image in detail")
print(description)
```

### Fine-tuning Management
```python
# List fine-tuning jobs
def list_fine_tuning_jobs():
    jobs = client.fine_tuning.jobs.list()
    for job in jobs.data:
        print(f"Job ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"Model: {job.fine_tuned_model}")
        print("---")

# Create a fine-tuning job
def create_fine_tuning_job(training_file_id, model="gpt-3.5-turbo"):
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 0.1
        }
    )
    return job.id

# Monitor fine-tuning job
def monitor_fine_tuning_job(job_id):
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Status: {job.status}")
    
    if job.status == "succeeded":
        print(f"Fine-tuned model: {job.fine_tuned_model}")
    elif job.status == "failed":
        print(f"Error: {job.error}")
```

---

## 11. Model Selection and Capabilities

### Model Comparison Table

| Model | Provider | Context Window | Best For | Function Calling | Vision | Pricing Tier |
|-------|----------|----------------|----------|------------------|---------|--------------|
| `gpt-4o` | OpenAI | 128K | General tasks, reasoning | ✅ | ✅ | High |
| `gpt-4o-mini` | OpenAI | 128K | Fast, lightweight tasks | ✅ | ✅ | Low |
| `gpt-4-turbo` | OpenAI | 128K | Complex reasoning | ✅ | ✅ | High |
| `gpt-3.5-turbo` | OpenAI | 16K | Cost-efficient chat | ✅ | ❌ | Low |
| `deepseek-chat` | DeepSeek | 32K | Math/code optimization | ✅ | ❌ | Medium |
| `llama-3-70b` | Groq | 8K | High-quality open model | ✅ | ❌ | Medium |
| `mistral-large` | Together AI | 32K | Multilingual tasks | ✅ | ❌ | Medium |
| `claude-3-opus` | Anthropic | 200K | Long-form content | ✅ | ✅ | High |

### Dynamic Model Selection
```python
def select_optimal_model(task_type, budget="medium"):
    models = {
        "creative_writing": {
            "high": "gpt-4o",
            "medium": "gpt-4o-mini",
            "low": "gpt-3.5-turbo"
        },
        "code_generation": {
            "high": "gpt-4o",
            "medium": "deepseek-chat",
            "low": "gpt-3.5-turbo"
        },
        "data_analysis": {
            "high": "gpt-4o",
            "medium": "gpt-4o-mini",
            "low": "gpt-3.5-turbo"
        },
        "summarization": {
            "high": "claude-3-opus",
            "medium": "gpt-4o-mini",
            "low": "gpt-3.5-turbo"
        }
    }
    
    return models.get(task_type, {}).get(budget, "gpt-3.5-turbo")

# Usage
model = select_optimal_model("code_generation", "medium")
print(f"Selected model: {model}")
```

---

## 12. Production-Ready Patterns

### Multi-Provider Client Factory
```python
from typing import Optional, Dict, Any
import os

class AIClientFactory:
    def __init__(self):
        self.clients = {}
    
    def get_client(self, provider: str, **kwargs) -> OpenAI:
        if provider in self.clients:
            return self.clients[provider]
        
        if provider == "openai":
            client = OpenAI(
                api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
                **kwargs
            )
        elif provider == "deepseek":
            client = OpenAI(
                api_key=kwargs.get("api_key", os.getenv("DEEPSEEK_API_KEY")),
                base_url="https://api.deepseek.com/v1",
                **kwargs
            )
        elif provider == "groq":
            client = OpenAI(
                api_key=kwargs.get("api_key", os.getenv("GROQ_API_KEY")),
                base_url="https://api.groq.com/openai/v1",
                **kwargs
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.clients[provider] = client
        return client

# Usage
factory = AIClientFactory()
openai_client = factory.get_client("openai")
groq_client = factory.get_client("groq")
```

### Chat Session Management
```python
from typing import List, Dict, Optional
import json
from datetime import datetime

class ChatSession:
    def __init__(self, client: OpenAI, model: str = "gpt-4", system_prompt: Optional[str] = None):
        self.client = client
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.usage_stats = {"total_tokens": 0, "requests": 0}
        
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def chat(self, user_message: str, **kwargs) -> str:
        self.add_message("user", user_message)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        
        assistant_message = response.choices[0].message.content
        self.add_message("assistant", assistant_message)
        
        # Track usage
        if hasattr(response, 'usage'):
            self.usage_stats["total_tokens"] += response.usage.total_tokens
            self.usage_stats["requests"] += 1
        
        return assistant_message
    
    def get_conversation_summary(self) -> Dict:
        return {
            "message_count": len(self.messages),
            "usage_stats": self.usage_stats,
            "model": self.model
        }
    
    def save_conversation(self, filename: str):
        conversation_data = {
            "messages": self.messages,
            "model": self.model,
            "usage_stats": self.usage_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
    
    def load_conversation(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.messages = data["messages"]
            self.model = data.get("model", self.model)
            self.usage_stats = data.get("usage_stats", {"total_tokens": 0, "requests": 0})

# Usage
session = ChatSession(
    client=client,
    model="gpt-4",
    system_prompt="You are a helpful coding assistant."
)

response1 = session.chat("How do I create a Python class?")
response2 = session.chat("Can you show me an example?")

print(session.get_conversation_summary())
session.save_conversation("coding_session.json")
```

### Caching Layer
```python
import hashlib
import json
from typing import Optional, Dict, Any
import time

class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
    
    def _get_cache_key(self, messages: List[Dict], model: str, **kwargs) -> str:
        # Create a hash of the request parameters
        cache_data = {
            "messages": messages,
            "model": model,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, messages: List[Dict], model: str, **kwargs) -> Optional[Any]:
        key = self._get_cache_key(messages, model, **kwargs)
        
        if key in self.cache:
            cached_data = self.cache[key]
            if time.time() - cached_data["timestamp"] < self.ttl:
                return cached_data["response"]
            else:
                del self.cache[key]
        
        return None
    
    def set(self, messages: List[Dict], model: str, response: Any, **kwargs):
        key = self._get_cache_key(messages, model, **kwargs)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }

class CachedOpenAIClient:
    def __init__(self, client: OpenAI, cache_ttl: int = 3600):
        self.client = client
        self.cache = ResponseCache(cache_ttl)
    
    def chat_completions_create(self, **kwargs):
        # Check cache first
        cached_response = self.cache.get(**kwargs)
        if cached_response:
            print("Cache hit!")
            return cached_response
        
        # Make API call
        response = self.client.chat.completions.create(**kwargs)
        
        # Cache the response
        self.cache.set(response=response, **kwargs)
        
        return response

# Usage
cached_client = CachedOpenAIClient(client, cache_ttl=1800)
response = cached_client.chat_completions_create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is Python?"}]
)
```

---

## 13. Legacy Completion API

### Text Completion (Deprecated but Still Supported)
```python
response = client.completions.create(
    model="text-davinci-003",
    prompt="Translate this into French: Hello, world!",
    max_tokens=60,
    temperature=0.7
)
print(response.choices[0].text)
```

> **Note**: Use `chat.completions.create` for all modern applications. The completions endpoint is legacy and not recommended for new projects.

---

## 14. Testing and Development

### Mock Client for Testing
```python
from unittest.mock import Mock, MagicMock

def create_mock_client():
    mock_client = Mock()
    
    # Mock chat completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a mock response"
    mock_response.usage.total_tokens = 50
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client

# Usage in tests
def test_chat_function():
    mock_client = create_mock_client()
    # Use mock_client in your tests
    response = mock_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}]
    )
    assert response.choices[0].message.content == "This is a mock response"
```

### Environment Configuration
```python
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

def get_openai_config(env: Environment) -> dict:
    configs = {
        Environment.DEVELOPMENT: {
            "model": "gpt-3.5-turbo",
            "max_tokens": 100,
            "temperature": 0.1,
            "timeout": 10
        },
        Environment.STAGING: {
            "model": "gpt-4o-mini",
            "max_tokens": 500,
            "temperature": 0.7,
            "timeout": 20
        },
        Environment.PRODUCTION: {
            "model": "gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30
        }
    }
    return configs[env]

# Usage
env = Environment(os.getenv("ENVIRONMENT", "development"))
config = get_openai_config(env)
```

---

## 15. Performance Optimization

### Batch Processing
```python
import asyncio
from typing import List, Dict, Any

async def process_batch(client: AsyncOpenAI, requests: List[Dict[str, Any]], batch_size: int = 5):
    results = []
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        
        tasks = []
        for request in batch:
            task = client.chat.completions.create(**request)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)
        
        # Small delay between batches to respect rate limits
        if i + batch_size < len(requests):
            await asyncio.sleep(0.1)
    
    return results

# Usage
async def main():
    client = AsyncOpenAI()
    
    requests = [
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Count to {i}"}],
            "max_tokens": 50
        }
        for i in range(1, 21)  # 20 requests
    ]
    
    results = await process_batch(client, requests, batch_size=5)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i} failed: {result}")
        else:
            print(f"Request {i}: {result.choices[0].message.content[:50]}...")

asyncio.run(main())
```

---

## 16. Security Best Practices

### API Key Management
```python
import os
from cryptography.fernet import Fernet

class SecureAPIKeyManager:
    def __init__(self, key_file: str = ".openai_key"):
        self.key_file = key_file
        self.cipher_suite = Fernet(self._get_or_create_key())
    
    def _get_or_create_key(self) -> bytes:
        key_file = ".secret_key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def store_api_key(self, api_key: str):
        encrypted_key = self.cipher_suite.encrypt(api_key.encode())
        with open(self.key_file, 'wb') as f:
            f.write(encrypted_key)
    
    def get_api_key(self) -> str:
        if not os.path.exists(self.key_file):
            raise ValueError("API key not found. Please store it first.")
        
        with open(self.key_file, 'rb') as f:
            encrypted_key = f.read()
        
        decrypted_key = self.cipher_suite.decrypt(encrypted_key)
        return decrypted_key.decode()

# Usage
key_manager = SecureAPIKeyManager()
# key_manager.store_api_key("your-actual-api-key")
api_key = key_manager.get_api_key()
```

### Input Sanitization
```python
import re
from typing import List

class InputSanitizer:
    def __init__(self):
        # Patterns for potentially dangerous content
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'data:text/html',            # Data URLs
            r'\\x[0-9a-fA-F]{2}',        # Hex escape sequences
        ]
    
    def sanitize_message(self, message: str) -> str:
        # Remove dangerous patterns
        for pattern in self.dangerous_patterns:
            message = re.sub(pattern, '', message, flags=re.IGNORECASE | re.DOTALL)
        
        # Limit message length
        if len(message) > 4000:
            message = message[:4000] + "... [truncated]"
        
        return message.strip()
    
    def sanitize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        sanitized = []
        for msg in messages:
            sanitized_msg = msg.copy()
            if 'content' in sanitized_msg:
                sanitized_msg['content'] = self.sanitize_message(sanitized_msg['content'])
            sanitized.append(sanitized_msg)
        return sanitized

# Usage
sanitizer = InputSanitizer()
safe_messages = sanitizer.sanitize_messages([
    {"role": "user", "content": "Tell me about <script>alert('xss')</script> Python"}
])
```

---

## 17. Monitoring and Logging

### Comprehensive Logging
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class OpenAILogger:
    def __init__(self, log_file: str = "openai_requests.log"):
        self.logger = logging.getLogger("openai_client")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_request(self, model: str, messages: List[Dict], **kwargs):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            "model": model,
            "message_count": len(messages),
            "parameters": kwargs
        }
        self.logger.info(json.dumps(log_data))
    
    def log_response(self, response, execution_time: float):
        usage = getattr(response, 'usage', None)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "response",
            "execution_time": execution_time,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            } if usage else None,
            "finish_reason": response.choices[0].finish_reason if response.choices else None
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        self.logger.error(json.dumps(log_data))

# Usage
logger = OpenAILogger()

def logged_chat_completion(messages, model="gpt-4", **kwargs):
    import time
    
    logger.log_request(model, messages, **kwargs)
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        execution_time = time.time() - start_time
        logger.log_response(response, execution_time)
        return response
        
    except Exception as e:
        logger.log_error(e, {"model": model, "messages": messages})
        raise
```

---

## 18. References & Resources

### Official Documentation
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)

### Provider-Specific Documentation
- [DeepSeek API Documentation](https://platform.deepseek.com/docs)
- [Groq API Documentation](https://console.groq.com/docs)
- [Together AI Documentation](https://docs.together.ai/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

### Integration Frameworks
- [LangChain OpenAI Integration](https://python.langchain.com/docs/integrations/llms/openai/)
- [LlamaIndex OpenAI Integration](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai/)
- [Haystack OpenAI Integration](https://docs.haystack.deepset.ai/docs/openai-generators)

### Development Tools
- [OpenAI Playground](https://platform.openai.com/playground)
- [Tokenizer Tool](https://platform.openai.com/tokenizer)
- [Model Comparison Tool](https://openai.com/api/pricing/)

---

## 19. Migration Guides

### From v0 to v1 SDK
```python
# Old SDK (v0)
import openai
openai.api_key = "sk-..."
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# New SDK (v1+)
from openai import OpenAI
client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Best Practices Summary
1. **Always use environment variables** for API keys
2. **Implement proper error handling** with retries
3. **Monitor usage and costs** with tracking systems
4. **Use structured outputs** for reliable data extraction
5. **Cache responses** when appropriate to reduce costs
6. **Sanitize user inputs** to prevent injection attacks
7. **Use async clients** for concurrent requests
8. **Implement rate limiting** to avoid quota issues
9. **Log all interactions** for debugging and monitoring
10. **Test with mock clients** during development

---

This comprehensive guide provides everything developers need to build production-ready applications with the OpenAI Python SDK. Use it as a reference for building robust, scalable, and secure AI-powered applications across multiple providers and use cases.
import asyncio
import os
from langchain.chat_models import init_chat_model
# ========================================================================
# Models
# ========================================================================

# Initialize a model
os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model(model="gpt-4o-mini")

response = model.invoke("Why do parrots talk?")

# ========================================================================
# Parameters
# ========================================================================
model = init_chat_model(
  model="claude-sonnet-4-5-20250929",
  temperature=0.5,
  timeout=30,
  max_tokens=1000,
)

# ========================================================================
# Invocation
# ========================================================================

# 1. invoke
response = model.invoke("Why do parrots have colorful feathers?") # single message prompt

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = model.invoke(conversation) # list of messages

# Message objects
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
  SystemMessage("You are a helpful assistant that translates English to French."),
  HumanMessage("Translate: I love programming."),
  AIMessage("J'adore la programmation."),
  HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)

# 2. stream

# Basic text streaming
for chunk in model.stream("Why do parrots have colorful feathers?"):
  print(chunk.text, end="|", flush=True)

# Stream tool calls, reasoning and other content
for chunk in model.stream("What color is the sky?"):
  for block in chunk.content_blocks:
    if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
      print(f"Reasoning: {reasoning}")
    elif block["type"] == "tool_call_chunk":
      print(f"Tool call chunk: {block}")
    elif block["type"] == "text":
      print(block["text"])
    else:
      ...

# Construct an AIMessage
full = None # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
  full = chunk if full is None else full + chunk

  print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

print(full.content_blocks)
# [{"type": "text", "text": "The sky is typically blue..."}]

# Advanced streaming topics
# Streaming events
async def stream_events():
  async for event in model.astream_events("Hello"):
    if event["event"] == "on_chat_model_start":
      print(f"Input: {event['data']['input']}")
    elif event["event"] == "on_chat_model_stream":
      print(f"Token: {event['data']['chunk'].text}")
    elif event["event"] == "on_chat_model_end":
      print(f"Full message: {event['data']['output'].text}")

asyncio.run(stream_events())

# 3. batch
responses = model.batch(
  [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
  ],
  config={
    'max_concurrency': 5 # Limit to 5 parallel calls
  }
)

for response in responses: # receive output at the end
  print(response)

for response in model.batch_as_completed([ # receive output for individual input
  "Why do parrots have colorful feathers?",
  "How do airplanes fly?",
  "What is quantum computing?"
]):
  print(response)

# ========================================================================
# Tool Calling
# ========================================================================
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
  """Get the weather at a location."""
  return f"It's sunny in {location}."

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in Boston?")

for tool_call in response.tool_calls:
  print(f"Tool: {tool_call['name']}")
  print(f"Args: {tool_call['args']}")

# Tool execution loop
# Bind (potentially multiple) tools to the model
model_with_tools = model.bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]

ai_msg = model_with_tools.invoke(messages)

messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
  # Execute the tool with the generated arguments
  tool_result = get_weather.invoke(tool_call)
  messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = model_with_tools.invoke(messages)

print(final_response.text)

# Forcing tool calls
model_with_tools = model.bind_tools(["tool_1"], tool_choice="any")

# Parallel tool calls
response = model_with_tools.invoke(
    "What's the weather in Boston and Tokyo?"
)

# The model may generate multiple tool calls
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]

# Execute all tools (can be done in parallel with async)
results = []

for tool_call in response.tool_calls:
  if tool_call['name'] == 'get_weather':
    result = get_weather.invoke(tool_call)

  ...

  results.append(result)

# Streaming tool calls
for chunk in model_with_tools.stream(
  "What's the weather in Boston and Tokyo?"
):
  # Tool call chunks arrive progressively
  for tool_chunk in chunk.tool_call_chunks:
    if name := tool_chunk.get("name"):
      print(f"Tool: {name}")
    if id_ := tool_chunk.get("id"):
      print(f"ID: {id_}")
    if args := tool_chunk.get("args"):
      print(f"Args: {args}")

# ========================================================================
# Structured Output
# ========================================================================

# Pydantic models
from pydantic import BaseModel, Field

class Movie(BaseModel):
  """A movie with details."""
  title: str = Field(..., description="The title of the movie")
  year: int = Field(..., description="The year the movie was released")
  director: str = Field(..., description="The director of the movie")
  rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)

response = model_with_structure.invoke("Provide details about the movie Inception")

print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)

# TypedDict
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
  """A movie with details."""
  title: Annotated[str, ..., "The title of the movie"]
  year: Annotated[int, ..., "The year the movie was released"]
  director: Annotated[str, ..., "The director of the movie"]
  rating: Annotated[float, ..., "The movie's rating out of 10"]

model_with_structure = model.with_structured_output(MovieDict)

response = model_with_structure.invoke("Provide details about the movie Inception")

print(response)  # {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}

# JSON Schema
import json

json_schema = {
  "title": "Movie",
  "description": "A movie with details",
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "The title of the movie"
    },
    "year": {
      "type": "integer",
      "description": "The year the movie was released"
    },
    "director": {
      "type": "string",
      "description": "The director of the movie"
    },
    "rating": {
      "type": "number",
      "description": "The movie's rating out of 10"
    }
  },
  "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
  json_schema,
  method="json_schema"
)

response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # {'title': 'Inception', 'year': 2010, ...}

# Message output alongside parsed structure

model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("Provide details about the movie Inception")
response
# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }

# Nested Structured
class Actor(TypedDict):
  name: str
  role: str

class MovieDetails(TypedDict):
  title: str
  year: int
  cast: list[Actor]
  genres: list[Actor]
  budget: Annotated[float | None, ..., "Budget in millions USD"]

model_with_structure = model.with_structured_output(MovieDetails)

# ========================================================================
# Advanced topics
# ========================================================================

# Model Profiles
model.profile
# {
#   "max_input_tokens": 400000,
#   "image_inputs": True,
#   "reasoning_output": True,
#   "tool_calling": True,
#   ...
# }

# Updating or overwriting profile data
# Option 1 (quick fix)
custom_profile = {
    "max_input_tokens": 100_000,
    "tool_calling": True,
    "structured_output": True,
    # ...
}

model = init_chat_model(model="gpt-4o-mini", profile=custom_profile)

new_profile = model.profile | {"key": "value"}
model.model_copy(update={"profile": new_profile})

# Option 2 (fix data upstream)

# ========================================================================
# Multimodal
# ========================================================================
response = model.invoke("Create a picture of a cat")
print(response.content_blocks)
# [
#     {"type": "text", "text": "Here's a picture of a cat"},
#     {"type": "image", "base64": "...", "mime_type": "image/jpeg"},
# ]

# ========================================================================
# Reasoning
# ========================================================================

# Stream reasoning output
for chunk in model.stream("Why do parrots have colorful feathers?"):
  reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
  print(reasoning_steps if reasoning_steps else chunk.text)

# Complete reasoning output
response = model.invoke("Why do parrots have colorful feathers?")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))

# ========================================================================
# Local Models
# ========================================================================

# Run modal on local machine e.g. Ollama

# ========================================================================
# Prompt caching
# ========================================================================

# Implicit prompt caching
# Providers will automatically pass on cost savings if a request hits a cache

# Explicit prompt caching
# Providers allow you to manually indicate cache points for greater control or to guarantee cost savings

# ========================================================================
# Server-side tool use
# ========================================================================
model = init_chat_model(model="gpt-4.1-mini")

tool = {"type": "web_search"}

model_with_tools = model.bind_tools([tool])

response = model_with_tools.invoke("What was a positive news story from today?")
response.content_blocks

# ========================================================================
# Rate Limiting
# ========================================================================

# Initialize and use a rate limiter
from langchain.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
  requests_per_second=0.1, # 1 request every 10s
  check_every_n_seconds=0.1, # Check every 100ms whether allowed to make a request
  max_bucket_size=10, # Controls the maximum burst size.
)

model = init_chat_model(
  model="gpt-5",
  model_provider="openai",
  rate_limiter=rate_limiter
)

# ========================================================================
# Base URL or Proxy
# ========================================================================

# Base URL
model = init_chat_model(
  model="MODEL_NAME",
  model_provider="openai",
  base_url="BASE_URL",
  api_key="API_KEY"
)

# Proxy Configuration
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
  model="gpt-4o-mini",
  openai_proxy="http://proxy.example.com:8080"
)

# ========================================================================
# Log probabilities
# ========================================================================

model = init_chat_model(
  model="gpt-4o-mini",
  model_provider="openai"
).bind(logprobs=True)

response = model.invoke("Why do parrots talk?")
print(response.response_metadata["logprobs"])

# ========================================================================
# Token Usage
# ========================================================================

# Callback Handler
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model(model="gpt-4o-mini")
model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

callback = UsageMetadataCallbackHandler()
result_1 = model_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = model_2.invoke("Hello", config={"callbacks": [callback]})
callback.usage_metadata

# ========================================================================
# Invocation Config
# ========================================================================

response = model.invoke(
  "Tell me a joke",
  config={
    "run_name": "joke_generation",
    "tags": ["humor", "demo"],
    "metadata": {"user_id": "123"},
    "callbacks": ["my_callback_handler"]
  }
)

# ========================================================================
# Configurable Models
# ========================================================================

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
  "what's your name",
  config={"configurable": {"model": "gpt-5-nano"}}
)

configurable_model.invoke(
  "what's your name",
  config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},  # Run with Claude
)

# Configurable model with default values
first_model = init_chat_model(
  model="gpt-4o-mini",
  temperature=0,
  configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
  config_prefix="first",
)

first_model.invoke("what's your name")

first_model.invoke(
  "what's your name",
  config={
    "configurable": {
      "first_model": "claude-sonnet-4-5-20250929",
      "first_temperature": 0.5,
      "first_max_tokens": 100,
    }
  },
)

# Using a configurable model declaratively
class GetWeather(BaseModel):
  """Get the current weather in a given location"""

  location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

class GetPopulation(BaseModel):
  """Get the current population in a given location"""

  location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

model = init_chat_model(temperature=0)
model_with_tools = model.bind_tools([GetWeather, GetPopulation])

model_with_tools.invoke(
  "what's bigger in 2024 LA or NYC", config={"configurable": {"model": "gpt-4.1-mini"}}
).tool_calls

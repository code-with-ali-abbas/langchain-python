# ================================================================================
# Messages
# ================================================================================

# Basic Usage
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(model="gpt-5-nano")

system_message = SystemMessage("You are a helpful assistant.")
human_message = HumanMessage("Hello, how are you?")

messages = [system_message, human_message]
response = model.invoke(messages) # Returns AIMessage

# Text Prompts
response = model.invoke("Write a haiku about spring")

# Message Prompts
messages = [
  SystemMessage("You are a poetry expert"),
  HumanMessage("Write a haiku about spring"),
  AIMessage("Cherry blossoms bloom...")
]

response = model.invoke(messages)

# Dictionary Format
messages = [
  {"role": "system", "content": "You are a poetry expert"},
  {"role": "user", "content": "Write a haiku about spring"},
  {"role": "assistant", "content": "Cherry blossoms bloom..."}
]

response = model.invoke(messages)

# ================================================================================
# Message Types
# ================================================================================

# System Message
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]

response = model.invoke(messages)

# Detailed persona
system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)

# Human Message

# Message object
response = model.invoke([
  HumanMessage("What is machine learning?")
])

# String shortcut
response = model.invoke("What is machine learning?")

# Message Metadata
human_message = HumanMessage(
  content="Hello!",
  name="alice", # Optional: identify different users
  id="msg_123", # Optional: unique identifier for tracing
)

# AI Message
response = model.invoke("Explain AI")
print(type(response)) # <class 'langchain.messages.AIMessage'>

# Create an AI message manually (e.g., for conversation history)
ai_msg = AIMessage("I'd be happy to help you with that question!")

messages = [
  SystemMessage("You are a helpful assistant"),
  HumanMessage("Can you help me?"),
  ai_msg,  # Insert as if it came from the model
  HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)

# Attributes
# 1. text: string
# 2. content: string | dict[]
# 3. content_blocks: ContentBlock[]
# 4. tool_calls: dict[] | None
# 5. id: string
# 6. usage_metadata: dict | None
# 7. response_metadata: ResponseMetadata | None

# Tool Calls
model = init_chat_model("gpt-5-nano")

def get_weather(location: str) -> str:
  """Get the weather at a location."""
  ...

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
  print(f"Tool: {tool_call['name']}")
  print(f"Args: {tool_call['args']}")
  print(f"ID: {tool_call['id']}")

# Token Usage
model = init_chat_model("gpt-5-nano")
response = model.invoke("Hello!")
response.usage_metadata

# {
#   'input_tokens': 8,
#   'output_tokens': 304,
#   'total_tokens': 312,
#   'input_token_details': {'audio': 0, 'cache_read': 0},
#   'output_token_details': {'audio': 0, 'reasoning': 256}
# }

# Streaming and Chunks
chunks = []
full_message = None

for chunk in model.stream("Hi"):
  chunks.append(chunk)
  print(chunk.text)
  full_message = chunk if full_message is None else full_message + chunk

# ToolMessage
from langchain.messages import ToolMessage

# After a model makes a tool call
# (Here, we demonstrate manually creating the messages for brevity)
ai_message = AIMessage(
  content=[],
  tool_calls=[{
    "name": "get_weather",
    "args": {"location": "San Francisco"},
    "id": "call_123"
  }]
)

# Execute tool and create result message
weather_result = "Sunny, 72°F"

tool_message = ToolMessage(
  content=weather_result,
  tool_call_id="call_123"  # Must match the call ID
)

messages = [
  HumanMessage("What's the weather in San Francisco?"),
  ai_message,  # Model's tool call
  tool_message,  # Tool execution result
]

response = model.invoke(messages)

# Sent to model
message_content = "It was the best of times, it was the worst of times."

# Artifact available downstream
artifact = {"document_id": "doc_123", "page": 0}

tool_message = ToolMessage(
  content=message_content,
  tool_call_id="call_123",
  name="search_books",
  artifact=artifact,
)

# Attributes
# 1. content: string required
# 2. tool_call_id: string required
# 3. name: string required
# 4. artifact: dict

# ================================================================================
# Message content
# ================================================================================

human_message = HumanMessage("Hello, how are you?")

human_message = HumanMessage(content=[
  {"type": "text", "text": "Hello, how are you?"},
  {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

human_message = HumanMessage(content=[
  {"type": "text", "text": "Hello, how are you?"},
  {"type": "image", "url": "https://example.com/image.jpg"},
])

# Standard Content Blocks
# Anthropic
message = AIMessage(
  content=[
    {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
    {"type": "text", "text": "..."},
  ],
  response_metadata={"model_provider": "anthropic"}
)

message.content_blocks
# [
#   {'type': 'reasoning', 'reasoning': '...', 'extras': {'signature': 'WaUjzkyp...'}},
#   {'type': 'text', 'text': '...'}
# ]

# OpenAI
message = AIMessage(
  content=[
    {
      "type": "reasoning",
      "id": "rs_abc123",
      "summary": [
        {"type": "summary_text", "text": "summary 1"},
        {"type": "summary_text", "text": "summary 2"},
      ],
    },
    {"type": "text", "text": "...", "id": "msg_abc123"},
  ],
  response_metadata={"model_provider": "openai"}
)

message.content_blocks

[
  {'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 1'},
  {'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 2'},
  {'type': 'text', 'text': '...', 'id': 'msg_abc123'}
]

# Serializing standard content
model = init_chat_model("gpt-5-nano", output_version="v1")

# ================================================================================
# Multimodal
# ================================================================================

# ****************************************************************************
# Image Input
# ****************************************************************************

# From URL
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this image."},
    {"type": "image", "url": "https://example.com/path/to/image.jpg"},
  ]
}

# From base64 data
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this image."},
    {
      "type": "image",
      "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
      "mime_type": "image/jpeg",
    },
  ]
}

# From provider-managed File ID
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this image."},
    {"type": "image", "file_id": "file-abc123"},
  ]
}

# ****************************************************************************
# PDF Document Input
# ****************************************************************************

# From URL
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this document."},
    {"type": "file", "url": "https://example.com/path/to/document.pdf"},
  ]
}

# From base64 data
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this document."},
    {
      "type": "file",
      "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
      "mime_type": "application/pdf",
    },
  ]
}

# From provider-managed File ID
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this document."},
    {"type": "file", "file_id": "file-abc123"},
  ]
}

# ****************************************************************************
# Audio Input
# ****************************************************************************

# From base64 data
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this audio."},
    {
      "type": "audio",
      "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
      "mime_type": "audio/wav",
    },
  ]
}

# From provider-managed File ID
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this audio."},
    {"type": "audio", "file_id": "file-abc123"},
  ]
}

# ****************************************************************************
# Video Input
# ****************************************************************************

# From base64 data
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this video."},
    {
      "type": "video",
      "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
      "mime_type": "video/mp4",
    },
  ]
}

# From provider-managed File ID
message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this video."},
    {"type": "video", "file_id": "file-abc123"},
  ]
}

# ===========================================================================
# Streaming Overview
# ===========================================================================

# 1. Agent Progress
from langchain.agents import create_agent

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  return f"It's always sunny in {city}!"

agent = create_agent(
  model="gpt-5-nano",
  tools=[get_weather],
)

for chunk in agent.stream(
  {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
  stream_mode="updates"
):
  for step, data in chunk.items():
    print(f"step: {step}")
    print(f"content: {data['messages'][-1].content_blocks}")

# 2. LLM Tokens

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  return f"It's always sunny in {city}!"

agent = create_agent(
  model="gpt-5-nano",
  tools=[get_weather],
)

for token, metadata in agent.stream(
  {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
  stream_mode="messages"
):
  print(f"node: {metadata['langgraph_node']}")
  print(f"content: {token.content_blocks}")
  print("\n")

# 3. Custom Updates
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  writer = get_stream_writer()
  # stream any arbitrary data
  writer(f"Looking up data for city: {city}")
  writer(f"Acquired data for city: {city}")
  return f"It's always sunny in {city}!"

agent = create_agent(
  model="claude-sonnet-4-5-20250929",
  tools=[get_weather],
)

for chunk in agent.stream(
  {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
  stream_mode="custom"
):
  print(chunk)

# 4. Stream Multiple Modes

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  writer = get_stream_writer()
  writer(f"Looking up data for city: {city}")
  writer(f"Acquired data for city: {city}")
  return f"It's always sunny in {city}!"

agent = create_agent(
  model="gpt-5-nano",
  tools=[get_weather],
)

for stream_mode, chunk in agent.stream(
  {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
  stream_mode=["updates", "custom"]
):
  print(f"stream_mode: {stream_mode}")
  print(f"content: {chunk}")
  print("\n")

# ===========================================================================
# Common Patterns
# ===========================================================================

# Streaming tool calls
from typing import Any
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  return f"It's always sunny in {city}!"

agent = create_agent("openai:gpt-5.2", tools=[get_weather])

def _render_message_chunk(token: AIMessageChunk) -> None:
  if token.text:
    print(token.text, end="|")
  if token.tool_call_chunks:
    print(token.tool_call_chunks)

def _render_completed_message(message: AnyMessage) -> None:
  if isinstance(message, AIMessage) and message.tool_calls:
    print(f"Tool calls: {message.tool_calls}")
  if isinstance(message, ToolMessage):
    print(f"Tool response: {message.content_blocks}")

input_message = {"role": "user", "content": "What is the weather in Boston?"}

for stream_mode, data in agent.stream(
  {"messages": [input_message]},
  stream_mode=["messages", "updates"]
):
  if stream_mode == "messages":
    token, metadata = data
    if isinstance(token, AIMessageChunk):
      _render_message_chunk(token)
  
  if stream_mode == "updates":
    for source, update in data.items():
      if source in ("model", "tools"):
        _render_completed_message(update["messages"][-1])

# Accessing completed messages
from typing import Literal
from langgraph.runtime import Runtime
from langchain.agents.middleware import after_agent, AgentState
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

class ResponseSafety(BaseModel):
  """Evaluate a response as safe or unsafe."""
  evaluation: Literal["safe", "unsafe"]

safety_model = init_chat_model("openai:gpt-5.2")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  """Model-based guardrail: Use an LLM to evaluate response safety."""

  stream_writer = get_stream_writer()

  if not state["messages"]:
    return None
  
  last_message = state["messages"][-1]

  if not isinstance(last_message, AIMessage):
    return None
  
  # Use another model to evaluate safety
  model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
  result = model_with_tools.invoke(
      [
        {
          "role": "system",
          "content": "Evaluate this AI response as generally safe or unsafe."
        },
        {
          "role": "user",
          "content": f"AI response: {last_message.text}"
        }
      ]
    )
  
  stream_writer(result)

  tool_call = result.tool_calls[0]
  if tool_call["args"]["evaluation"] == "unsafe":
    last_message.content = "I cannot provide that response. Please rephrase your request."

  return None

# We can then incorporate this middleware into our agent and include its custom stream events:

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  return f"It's always sunny in {city}!"

agent = create_agent(
  model="openai:gpt-5.2",
  tools=[get_weather],
  middleware=[safety_guardrail]
)

def _render_message_chunk(token: AIMessageChunk) -> None:
  if token.text:
    print(token.text, end="|")
  if token.tool_call_chunks:
    print(token.tool_call_chunks)

def _render_completed_message(message: AnyMessage) -> None:
  if isinstance(message, AIMessage) and message.tool_calls:
    print(f"Tool calls: {message.tool_calls}")
  if isinstance(message, ToolMessage):
    print(f"Tool response: {message.content_blocks}")

input_message = {"role": "user", "content": "What is the weather in Boston?"}

for stream_mode, data in agent.stream(
  {"messages": [input_message]},
  stream_mode=["messages", "updates", "custom"]
):
  if stream_mode == "messages":
    token, metadata = data
    if isinstance(token, AIMessageChunk):
      _render_message_chunk(token)
  
  if stream_mode == "updates":
    for source, update in data.items():
      if source in ("model", "tools"):
        _render_completed_message(update["messages"][-1])

  if stream_mode == "custom":
    print(f"Tool calls: {data.tool_calls}")

# Streaming with human-in-the-loop
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  return f"It's always sunny in {city}!"

checkpointer = InMemorySaver()

agent = create_agent(
  model="openai:gpt-5.2",
  tools=[get_weather],
  middleware=[
    HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})
  ],
  checkpointer=checkpointer,
)

def _render_message_chunk(token: AIMessageChunk) -> None:
  if token.text:
    print(token.text, end="|")
  if token.tool_call_chunks:
    print(token.tool_call_chunks)

def _render_completed_message(message: AnyMessage) -> None:
  if isinstance(message, AIMessage) and message.tool_calls:
    print(f"Tool calls: {message.tool_calls}")
  if isinstance(message, ToolMessage):
    print(f"Tool response: {message.content_blocks}")

def _render_interrupt(interrupt: Interrupt) -> None:
  interrupts = interrupt.value

  for request in interrupts["action_requests"]:
    print(request["description"])

input_message = {
  "role": "user",
  "content": (
    "Can you look up the weather in Boston and San Francisco?"
  ),
}

config = {"configurable": {"thread_id": "some_id"}}
interrupts = []

for stream_mode, data in agent.stream(
  {"messages": [input_message]},
  config=config,  
  stream_mode=["messages", "updates"],
):
  if stream_mode == "messages":
    token, metadata = data
    if isinstance(token, AIMessageChunk):
       _render_message_chunk(token)
  if stream_mode == "updates":
    for source, update in data.items():
      if source in ("model", "tools"):
        _render_completed_message(update["messages"][-1])
      if source == "__interrupt__":
        interrupts.extend(update)
        _render_interrupt(update[0])

def _get_interrupt_decisions(interrupt: Interrupt) -> list[dict]:
  return [
    {
      "type": "edit",
      "edited_action": {
        "name": "get_weather",
        "args": {"city": "Boston, U.K."},
      },
    }
    if "boston" in request["description"].lower()
    else {"type": "approve"}
    for request in interrupt.value["action_requests"]
  ]

decisions = {}
for interrupt in interrupts:
  decisions[interrupt.id] = {
    "decisions": _get_interrupt_decisions(interrupt)
  }

decisions

interrupts = []
for stream_mode, data in agent.stream(
  {"messages": [input_message]},
  Command(resume=decisions),
  config=config,
  stream_mode=["messages", "updates"],
):
  if stream_mode == "messages":
    token, metadata = data
    if isinstance(token, AIMessageChunk):
       _render_message_chunk(token)
  if stream_mode == "updates":
    for source, update in data.items():
      if source in ("model", "tools"):
        _render_completed_message(update["messages"][-1])
      if source == "__interrupt__":
        interrupts.extend(update)
        _render_interrupt(update[0])

# Streaming from sub-agents

def get_weather(city: str) -> str:
  """Get weather for a given city."""
  return f"It's always sunny in {city}!"

weather_model = init_chat_model(
  model="openai:gpt-5.2",
  tags=["weather_sub_agent"],
)

weather_agent = create_agent(model=weather_model, tools=[get_weather])

def call_weather_agent(query: str) -> str:
  """Query the weather agent."""
  result = weather_agent.invoke({
    "messages": [{"role": "user", "content": query}]
  })
  return result["messages"][-1].text

supervisor_model = init_chat_model(
  model="openai:gpt-5.2",
  tags=["supervisor"],
)

agent = create_agent(model=supervisor_model, tools=[call_weather_agent])

input_message = {"role": "user", "content": "What is the weather in Boston?"}

current_agent = None

for _, stream_mode, data in agent.stream(
  {"messages": [input_message]},
  stream_mode=["messages", "updates"],
  subgraphs=True,  
):
  if stream_mode == "messages":
    token, metadata = data
    if tags := metadata.get("tags", []):  
      this_agent = tags[0]  
      if this_agent != current_agent:  
        print(f"🤖 {this_agent}: ")  
        current_agent = this_agent  
    if isinstance(token, AIMessage):
      _render_message_chunk(token)
  if stream_mode == "updates":
    for source, update in data.items():
      if source in ("model", "tools"):
        _render_completed_message(update["messages"][-1])

# ===========================================================================
# Disable streaming
# ===========================================================================
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    streaming=False
)

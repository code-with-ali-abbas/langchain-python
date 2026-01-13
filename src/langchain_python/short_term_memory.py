# ===================================================================
# Short Term Memory
# ===================================================================

# Usage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
  model="gpt-5",
  tools=["get_user_info"],
  checkpointer=InMemorySaver(),
)

agent.invoke(
  {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
  {"configurable": {"thread_id": "1"}}
)

# In production: use a checkpointer backed by a database:
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
  checkpointer.setup() # auto create tables in PostgresSql
  agent = create_agent(
    model="gpt-5",
    tools=["get_user_info"],
    checkpointer=checkpointer,
  )

# ===================================================================
# Customizing agent memory
# ===================================================================

from langchain.agents import AgentState

class CustomAgentState(AgentState):
  user_id: str
  preferences: dict

agent = create_agent(
  model="gpt-5",
  tools=["get_user_info"],
  state_schema=CustomAgentState,
  checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
  {
    "messages": [{"role": "user", "content": "Hello"}],
    "user_id": "user_123",  
    "preferences": {"theme": "dark"}  
  },
  {"configurable": {"thread_id": "1"}}
)

# ===================================================================
# Common Patterns
# ===================================================================

# 1. Trim Message
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

@before_model
def trim_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  """Keep only the last few messages to fit context window."""
  messages = state["messages"]

  if len(messages) <= 3:
    return None
  
  first_message = messages[0]
  recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
  new_messages = [first_message] + recent_messages

  return {
    "messages": [
      RemoveMessage(id=REMOVE_ALL_MESSAGES),
      *new_messages
    ]
  }

agent = create_agent(
  model="gpt-5",
  tools=["your_tools_here"],
  middleware=[trim_message],
  checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()

"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""

# 2. Delete Messages

# To remove specific messages:
def delete_messages(state):
  messages = state["messages"]

  if len(messages) > 2:
    # remove the earliest two messages
    return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
  
# To remove all messages:
def delete_messages(state):
  return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

# -----------------------------------------------------------------------------------------
from langchain.agents.middleware import after_model

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
  """Remove old messages to keep conversation manageable."""
  messages = state["messages"]

  if len(messages) > 2:
    return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
  return None

agent = create_agent(
  model="gpt-5",
  tools=[],
  system_prompt="Please be concise and to the point",
  middleware=[delete_old_messages],
  checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
  {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
  config,
  stream_mode="values",
):
  print([(message.type, message.content) for message in event["messages"]])

for event in agent.stream(
  {"messages": [{"role": "user", "content": "what's my name?"}]},
  config,
  stream_mode="values",
):
  print([(message.type, message.content) for message in event["messages"]])

# 3. Summarize Messages
from langchain.agents.middleware import SummarizationMiddleware

checkpointer = InMemorySaver()

agent = create_agent(
  model="gpt-5",
  tools=[],
  checkpointer=checkpointer,
  middleware=[
    SummarizationMiddleware(
      model="gpt-4o-mini",
      trigger=("tokens", 4000),
      keep=("messages", 20)
    )
  ]
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()

"""
================================== Ai Message ==================================

Your name is Bob!
"""

# ===================================================================
# Access Memory
# ===================================================================

# You can access and modify the short-term memory (state) of an agent in several ways:
# Tools: Read short-term memory in a tool

from langchain.tools import tool, ToolRuntime

class CustomState(AgentState):
  user_id: str

@tool
def get_user_info(
  runtime: ToolRuntime
) -> str:
  """Look up user info."""
  user_id = runtime.state["user_id"]
  return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
  model="gpt-5",
  tools=[get_user_info],
  state_schema=CustomState
)

result = agent.invoke({"messages": "look up user information", "user_id": "user_123"})
print(result["messages"][-1].content)
# > User is John Smith.

# Write short-term memory from tools
from langgraph.types import Command
from pydantic import BaseModel
from langchain.messages import ToolMessage

class CustomState(AgentState):  
  user_name: str

class CustomContext(BaseModel):
  user_id: str

@tool
def update_user_info(
  runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
  """Look up and update user info."""
  user_id = runtime.context.user_id
  name = "John Smith" if user_id == "user_123" else "Unknown user"

  return Command(update={  
    "user_name": name,
    # update the message history
    "messages": [
      ToolMessage(
        "Successfully looked up user information",
        tool_call_id=runtime.tool_call_id
      )
    ]
  })

@tool
def greet(
  runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
  """Use this to greet the user once you found their info."""
  user_name = runtime.state.get("user_name", None)
  if user_name is None:
    return Command(update={
      "messages": [
        ToolMessage(
          "Please call the 'update_user_info' tool it will get and update the user's name.",
          tool_call_id=runtime.tool_call_id
        )
      ]
    })
  return f"Hello {user_name}!"

agent = create_agent(
  model="gpt-5-nano",
  tools=[update_user_info, greet],
  state_schema=CustomState, 
  context_schema=CustomContext,
)

agent.invoke(
  {"messages": [{"role": "user", "content": "greet the user"}]},
  context=CustomContext(user_id="user_123"),
)

# Prompt
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class CustomContext(TypedDict):
  user_name: str

def get_weather(city: str) -> str:
  """Get the weather in a city."""
  return f"The weather in {city} is always sunny!"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
  user_name = request.runtime.context["user_name"]
  system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
  return system_prompt

agent = create_agent(
  model="gpt-5-nano",
  tools=[get_weather],
  middleware=[dynamic_system_prompt],
  context_schema=CustomContext,
)

result = agent.invoke(
  {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
  context=CustomContext(user_name="John Smith"),
)

for msg in result["messages"]:
  msg.pretty_print()

# After model

@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
  """Remove messages containing sensitive words."""
  STOP_WORDS = ["password", "secret"]
  last_message = state["messages"][-1]
  if any(word in last_message.content for word in STOP_WORDS):
    return {"messages": [RemoveMessage(id=last_message.id)]}
  return None

agent = create_agent(
  model="gpt-5-nano",
  tools=[],
  middleware=[validate_response],
  checkpointer=InMemorySaver(),
)

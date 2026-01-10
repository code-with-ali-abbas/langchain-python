# =======================================================================================
# 1. Static Models
# =======================================================================================
from langchain.agents import create_agent

tools = ["tool_1", "tool_2"]
agent = create_agent("openai:gpt-5", tools=tools)

# For more control over the model configuration, initialize a model instance directly using the provider package.
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
  model="gpt-5",
  temperature=0.1,
  max_tokens=1000,
  timeout=30,
)
agent = create_agent(model=model, tools=tools)

# =======================================================================================
# 2. Dynamic Model
# =======================================================================================
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
  """Choose model based on conversation complexity."""
  message_count = len(request.state['messages'])

  if message_count > 10:
    model = advanced_model
  else:
    model = basic_model

  return handler(request.override(model=model))

agent = create_agent(
  model=basic_model,
  tools=tools,
  middleware=[dynamic_model_selection]
)

# =======================================================================================
# Defining tools
# =======================================================================================
from langchain.tools import tool

@tool
def search(query: str) -> str:
  """Search for information."""
  return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
  """Get weather information for a location."""
  return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model=model, tools=[search, get_weather])

# =======================================================================================
# Tool Error Handling
# =======================================================================================
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
  """Handle tool execution errors with custom messages."""
  try:
    return handler(request)
  except Exception as e:
    # Return a custom error message to the model
    return ToolMessage(
      content=f"Tool error: Please check your input and try again. ({str(e)})",
      tool_call_id=request.tool_call["id"]
    )
  
agent = create_agent(
  model="gpt-4o",
  tools=[search, get_weather],
  middleware=[handle_tool_errors],
)

# =======================================================================================
# System Prompt
# =======================================================================================

# Using string
agent = create_agent(
  model="gpt-4o",
  tools=tools,
  system_prompt="You are a helpful assistant. Be concise and accurate."
)

# Using SystemMessage
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
  model="anthropic:claude-sonnet-4-5",
  system_prompt=SystemMessage(
    content=[
      {
        "type": "text",
        "text": "You are an AI assistant tasked with analyzing literary works.",
      },
      {
        "type": "text",
        "text": "<the entire contents of 'Pride and Prejudice'>",
        "cache_control": {"type": "ephemeral"}
      }
    ]
  )
)

result = literary_agent.invoke(
  {
    "messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]
  }
)

# =======================================================================================
# Dynamic System Prompt
# =======================================================================================
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt

class Context(TypedDict):
  user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
  """Generate system prompt based on user role."""

  user_role = request.runtime.context.get("user_role", "user")
  base_prompt = "You are a helpful assistant."

  if user_role == "expert":
    return f"{base_prompt} Provide detailed technical responses."
  elif user_role == "beginner":
    return f"{base_prompt} Provide concepts simply and avoid jargon."
  
  return base_prompt

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Web results for: {query}"

agent = create_agent(
  model="gpt-4o",
  tools=[web_search],
  middleware=[user_role_prompt],
  context_schema=Context
)

result = agent.invoke(
  {"messages": [{"role": "user", "context": "Explain machine learning"}]},
  context={"user_role": "expert"}
)

# =======================================================================================
# Invocation
# =======================================================================================

result = agent.invoke(
  {"messages": [{"role": "user", "context": "What's the weather in San Francisco?"}]},
)

# =======================================================================================
# Advanced concepts
# =======================================================================================
# Structured output
# =======================================================================================

# ToolStrategy
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
  name: str
  email: str
  phone: str

agent = create_agent(
  model="gpt-4o-mini",
  response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke(
  {"messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]}
)

result["structured_output"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')

# ProviderStrategy
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
  model="gpt-4o-mini",
  response_format=ProviderStrategy(ContactInfo)
)

# =======================================================================================
# Memory
# =======================================================================================

# Via Middleware (Preferred)
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

@tool
def tool_1(input: str) -> str:
    return f"Tool 1 received: {input}"

@tool
def tool_2(input: str) -> str:
    return f"Tool 2 received: {input}"

class CustomState(AgentState):
  user_preferences: dict

class CustomMiddleware(AgentMiddleware):
  state_schema = CustomState
  tools = [tool_1, tool_2]

  def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
    preferences = state.user_preferences
    runtime.logger.info(f"User preferences: {preferences}")
    return None
  
agent = create_agent(
  model="gpt-4o-mini",
  tools=tools,
  middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})

# Via state_schema
class CustomState(AgentState):
  user_preferences: dict

agent = create_agent(
  model="gpt-4o-mini",
  tools=[tool_1, tool_2],
  state_schema=CustomState
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})

# =======================================================================================
# Streaming
# =======================================================================================
for chunk in agent.stream({
  "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values"):
  # Each chunk contains the full state at that point
  latest_message = chunk["messages"][-1]

  if latest_message.content:
    print(f"Agent: {latest_message.content}")
  elif latest_message.tool_calls:
    print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

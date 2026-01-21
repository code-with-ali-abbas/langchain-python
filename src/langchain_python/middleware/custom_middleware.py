# ===========================================================================
# Custom Middleware
# ===========================================================================

# Node-style hooks

# Using Decorator
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  if len(state["messages"]) >= 50:
    return {
      "messages": [AIMessage("Conversation limit reached.")],
      "jump_to": "end"
    }
  return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  print(f"Model returned: {state['messages'][-1].content}")
  return None

# Using Class
from langchain.agents.middleware import AgentMiddleware, hook_config
class MessageLimitMiddleware(AgentMiddleware):
  def __init__(self, max_messages: int = 50):
    super().__init__()
    self.max_messages = max_messages

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
      if len(state["messages"]) == self.max_messages:
        return {
          "messages": [AIMessage("Conversation limit reached.")],
          "jump_to": "end"
        }
      return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
      print(f"Model returned: {state['messages'][-1].content}")
      return None
    
# Swap-style hooks
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

# Using Decorator
@wrap_model_call
def retry_model(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  for attempt in range(3):
    try:
      return handler(request)
    except Exception as e:
      if attempt == 2:
        raise
      print(f"Retry {attempt + 1}/3 after error: {e}")

# Using Class
class RetryMiddleware(AgentMiddleware):
  def __init__(self, max_retries: int = 3):
    super().__init__()
    self.max_retries = max_retries

  def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
  ) -> ModelResponse:
    for attempt in range(self.max_retries):
      try:
        return handler(request)
      except Exception as e:
        if attempt == self.max_retries - 1:
          raise
        print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")

# Create Middleware

# Decorator-based middleware
from langchain.agents import create_agent

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  print(f"About to call model with {len(state['messages'])} messages")
  return None

@wrap_model_call
def retry_model(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
  for attempt in range(3):
    try:
      return handler(request)
    except Exception as e:
      if attempt == 2:
        raise
      print(f"Retry {attempt + 1}/3 after error: {e}")

agent = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[log_before_model, retry_model]
)

# Class-based middleware
class LoggingMiddleware(AgentMiddleware):
  def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state['messages'])} messages")
    return None

  def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None

agent = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[LoggingMiddleware()],
)

# Custom state schema

# Using Decorator
from typing_extensions import NotRequired
from langchain.messages import HumanMessage

class CustomState(AgentState):
  model_call_count: NotRequired[int]
  user_id: NotRequired[str]

@before_model(state_schema=CustomState, can_jump_to=["end"])
def check_call_limit(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
  count = state.get("model_call_count", 0)
  if count > 10:
    return {"jump_to": "end"}
  return None

@after_model(state_schema=CustomState)
def increment_counter(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
  return {"model_call_count": state.get("model_call_count", 0) + 1}

agent = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[check_call_limit, increment_counter],
)

result = agent.invoke({
  "messages": [HumanMessage("Hello")],
  "model_call_count": 0,
  "user_id": "user-123",
})

# Using Class
class CustomState(AgentState):
  model_call_count: NotRequired[int]
  user_id: NotRequired[str]

class CallCounterMiddleware(AgentMiddleware[CustomState]):
  state_schema = CustomState

  def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
    count = state.get("model_call_count", 0)
    if count > 10:
      return {"jump_to": "end"}
    return None

  def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}
  
agent = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[CallCounterMiddleware()],
)

result = agent.invoke({
  "messages": [HumanMessage("Hello")],
  "model_call_count": 0,
  "user_id": "user-123",
})

# Execution order

# Agent jumps
# To exit early from middleware, return a dictionary with jump_to:

# Using Decorator
@after_model
@hook_config(can_jump_to=["end"])
def check_for_blocked(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  last_message = state["messages"][-1]
  if "BLOCKED" in last_message.content:
    return {
      "messages": [AIMessage("I cannot respond to that request.")],
      "jump_to": "end"
    }
  return None

# Using Class
class BlockedContentMiddleware(AgentMiddleware):
  @hook_config(can_jump_to=["end"])
  def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
      return {
        "messages": [AIMessage("I cannot respond to that request.")],
        "jump_to": "end"
      }
    return None
  
# Example: Dynamic model selection

# Using Decorator
from langchain.chat_models import init_chat_model

complex_model = init_chat_model("gpt-4o")
simple_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def dynamic_model(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
  if len(request.messages) > 10:
    model = complex_model
  else:
    model = simple_model
  return handler(request.override(model=model))

# Using Class
class DynamicModelMiddleware(AgentMiddleware):
  def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
  ) -> ModelResponse:
    if len(request.messages) > 10:
      model = complex_model
    else:
      model = simple_model
    return handler(request.override(model=model))
  
# Tool call monitoring
# Using Decorator
from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

@wrap_tool_call
def monitor_tool(
  request: ToolCallRequest,
  handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
  print(f"Executing tool: {request.tool_call['name']}")
  print(f"Arguments: {request.tool_call['args']}")
  try:
    result = handler(request)
    print(f"Tool completed successfully")
    return result
  except Exception as e:
    print(f"Tool failed: {e}")
    raise

# Using Class
class ToolMonitoringMiddleware(AgentMiddleware):
  def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
  ) -> ToolMessage | Command:
    print(f"Executing tool: {request.tool_call['name']}")
    print(f"Arguments: {request.tool_call['args']}")
    try:
      result = handler(request)
      print(f"Tool completed successfully")
      return result
    except Exception as e:
      print(f"Tool failed: {e}")
      raise

# Dynamically selecting tools

# Using Decorator
@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware to select relevant tools based on state/context."""
    # Select a small, relevant subset of tools based on state/context
    relevant_tools = select_relevant_tools(request.state, request.runtime)
    return handler(request.override(tools=relevant_tools))

agent = create_agent(
  model="gpt-4o",
  tools=["all_tools"],  # All available tools need to be registered upfront
  middleware=[select_tools],
)

# Using Class
class ToolSelectorMiddleware(AgentMiddleware):
  def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
  ) -> ModelResponse:
    """Middleware to select relevant tools based on state/context."""
    # Select a small, relevant subset of tools based on state/context
    relevant_tools = select_relevant_tools(request.state, request.runtime)
    return handler(request.override(tools=relevant_tools))

agent = create_agent(
  model="gpt-4o",
  tools=["all_tools"],  # All available tools need to be registered upfront
  middleware=[ToolSelectorMiddleware()],
)

# Working with system messages
# Using Decorator
from langchain.messages import SystemMessage

@wrap_model_call
def add_context(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
  # Always work with content blocks
  new_content = list(request.system_message.content_blocks) + [
      {"type": "text", "text": "Additional context."}
  ]
  new_system_message = SystemMessage(content=new_content)
  return handler(request.override(system_message=new_system_message))

# Using Class
class ContextMiddleware(AgentMiddleware):
  def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
  ) -> ModelResponse:
    # Always work with content blocks
    new_content = list(request.system_message.content_blocks) + [
      {"type": "text", "text": "Additional context."}
    ]
    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))

# Working with cache control (Anthropic)
# Using Decorator
@wrap_model_call
def add_cached_context(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
  # Always work with content blocks
  new_content = list(request.system_message.content_blocks) + [
    {
      "type": "text",
      "text": "Here is a large document to analyze:\n\n<document>...</document>",
      # content up until this point is cached
      "cache_control": {"type": "ephemeral"}
    }
  ]

  new_system_message = SystemMessage(content=new_content)
  return handler(request.override(system_message=new_system_message))

# Using Class
class CachedContextMiddleware(AgentMiddleware):
  def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
  ) -> ModelResponse:
    # Always work with content blocks
    new_content = list(request.system_message.content_blocks) + [
      {
        "type": "text",
        "text": "Here is a large document to analyze:\n\n<document>...</document>",
        "cache_control": {"type": "ephemeral"}  # This content will be cached
      }
    ]

    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))

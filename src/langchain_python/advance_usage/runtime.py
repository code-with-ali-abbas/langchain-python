# ===========================================================================
# Runtime
# ===========================================================================

# ---------------------------------------------
# Access
# ---------------------------------------------
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class Context:
  user_name: str

agent = create_agent(
  model="gpt-4o-mini",
  tools=[...],
  context_schema=Context
)

agent.invoke(
  {"messages": [{"role": "user", "content": "What's my name?"}]},
  context=Context(user_name="Alice")
)

# ---------------------------------------------
# Inside tools
# ---------------------------------------------
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
  user_id: str

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
  """Fetch the user's email preferences from the store."""
  user_id = runtime.context.user_id

  preferences: str = "The user prefers you to write a brief and polite email."

  if runtime.store:
    if memory := runtime.store.get(("users"), user_id):
      preferences = memory.value["preferences"]

  return preferences

# ---------------------------------------------
# Inside middleware
# ---------------------------------------------
from langchain.messages import AnyMessage
from langchain.agents import AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model, after_model
from langgraph.runtime import Runtime

@dataclass
class Context:
  user_name: str

# Dynamic prompts
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
  user_name = request.runtime.context.user_name  
  system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
  return system_prompt

# Before model hook
@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:  
  print(f"Processing request for user: {runtime.context.user_name}")  
  return None

# After model hook
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:  
  print(f"Completed request for user: {runtime.context.user_name}")  
  return None

agent = create_agent(
  model="gpt-5-nano",
  tools=[...],
  middleware=[dynamic_system_prompt, log_before_model, log_after_model],  
  context_schema=Context
)

agent.invoke(
  {"messages": [{"role": "user", "content": "What's my name?"}]},
  context=Context(user_name="John Smith")
)

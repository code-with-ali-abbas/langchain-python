# ===========================================================================
# Context Engineering
# ===========================================================================

# --------------------------------------
# System Prompt
# --------------------------------------

# Access message count or conversation context from state:
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
  message_count = len(request.messages)

  base = "You are a helpful assistant."

  if message_count > 10:
    base += "\nThis is a long conversation - be extra concise."

  return base

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[state_aware_prompt]
)

# Access user preferences from long-term memory:
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
  user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
  user_id = request.runtime.context.user_id

  # Read from Store: get user preferences
  store = request.runtime.store
  user_preferences = store.get(("preferences",), user_id)

  base = "You are a helpful assistant."

  if user_preferences:
      style = user_preferences.value.get("communication_style", "balanced")
      base += f"\nUser prefers {style} responses."

  return base

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[store_aware_prompt],
  context_schema=Context,
  store=InMemoryStore()
)

# Access user ID or configuration from Runtime Context:

@dataclass
class Context:
  user_role: str
  deployment_env: str

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
  # Read from Runtime Context: user role and environment
  user_role = request.runtime.context.user_role
  env = request.runtime.context.deployment_env

  base = "You are a helpful assistant."

  if user_role == "admin":
    base += "\nYou have admin access. You can perform all operations."
  elif user_role == "viewer":
    base += "\nYou have read-only access. Guide users to read operations only."

  if env == "production":
    base += "\nBe extra careful with any data modifications."

  return base

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[context_aware_prompt],
  context_schema=Context
)

# --------------------------------------
# Messages
# --------------------------------------

# Inject uploaded file context from State when relevant to current query:
from langchain.agents.middleware import wrap_model_call, ModelResponse
from typing import Callable

@wrap_model_call
def inject_file_context(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Inject context about files user has uploaded this session."""

  # Read from State: get uploaded files metadata
  uploaded_files = request.state.get("uploaded_files", [])

  if uploaded_files:
    # Build context about available files
    file_descriptions = []

    for file in uploaded_files:
      file_descriptions.append(
        f"- {file['name']} ({file['type']}): {file['summary']}"
      )

    file_context = f"""Files you have access to in this conversation:
      {chr(10).join(file_descriptions)}

      Reference these files when answering questions."""

    messages = [  
      *request.messages,
      {"role": "user", "content": file_context},
    ]

    request = request.override(messages=messages)

    return handler(request)
  
agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[inject_file_context]
)

# Inject user’s email writing style from Store to guide drafting:

@dataclass
class Context:
  user_id: str

@wrap_model_call
def inject_writing_style(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Inject user's email writing style from Store."""
  user_id = request.runtime.context.user_id

  # Read from Store: get user's writing style examples
  store = request.runtime.store
  writing_style = store.get(("writing_style",), user_id)

  if writing_style:
    style = writing_style.value

    # Build style guide from stored examples
    style_context = f"""Your writing style:
      - Tone: {style.get('tone', 'professional')}
      - Typical greeting: "{style.get('greeting', 'Hi')}"
      - Typical sign-off: "{style.get('sign_off', 'Best')}"
      - Example email you've written:
      {style.get('example_email', '')}"""
    
    # Append at end - models pay more attention to final messages
    messages = [
      *request.messages,
      {"role": "user", "content": style_context}
    ]

    request = request.override(messages=messages)  

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[inject_writing_style],
  context_schema=Context,
  store=InMemoryStore()
)

# Inject compliance rules from Runtime Context based on user’s jurisdiction:
@dataclass
class Context:
  user_jurisdiction: str
  industry: str
  compliance_frameworks: list[str]

@wrap_model_call
def inject_compliance_rules(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Inject compliance constraints from Runtime Context."""
  # Read from Runtime Context: get compliance requirements

  jurisdiction = request.runtime.context.user_jurisdiction
  industry = request.runtime.context.industry
  frameworks = request.runtime.context.compliance_frameworks

  # Build compliance constraints
  rules = []

  if "GDPR" in frameworks:
    rules.append("- Must obtain explicit consent before processing personal data")
    rules.append("- Users have right to data deletion")
  if "HIPAA" in frameworks:
    rules.append("- Cannot share patient health information without authorization")
    rules.append("- Must use secure, encrypted communication")
  if industry == "finance":
    rules.append("- Cannot provide financial advice without proper disclaimers")

  if rules:
      compliance_context = f"""Compliance requirements for {jurisdiction}: {chr(10).join(rules)}"""
      
      # Append at end - models pay more attention to final messages
      messages = [
        *request.messages,
        {"role": "user", "content": compliance_context}
      ]

      request = request.override(messages=messages)  

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[inject_compliance_rules],
  context_schema=Context
)

# --------------------------------------
# Tools
# --------------------------------------

from langchain.tools import tool

@tool(parse_docstring=True)
def search_orders(
  user_id: str,
  status: str,
  limit: int = 10
) -> str:
  """Search for user orders by status.

  Use this when the user asks about order history or wants to check
  order status. Always filter by the provided status.

  Args:
    user_id: Unique identifier for the user
    status: Order status: 'pending', 'shipped', or 'delivered'
    limit: Maximum number of results to return
  """

  # Implementation here
  pass

# Selecting tools

# Enable advanced tools only after certain conversation milestones:
@wrap_model_call
def state_based_tools(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Filter tools based on conversation State."""
  # Read from State: check if user has authenticated
  state = request.state  
  is_authenticated = state.get("authenticated", False)  
  message_count = len(state["messages"])

  if not is_authenticated:
    tools = [t for t in request.tools if t.name.startswith("public_")]
    request = request.override(tools=tools)
  elif message_count < 5:
    # Limit tools early in conversation
    tools = [t for t in request.tools if t.name != "advanced_search"]
    request = request.override(tools=tools)

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=["public_search", "private_search", "advanced_search"],
  middleware=[state_based_tools]
)

# Filter tools based on user preferences or feature flags in Store:
@dataclass
class Context:
  user_id: str

@wrap_model_call
def store_based_tools(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Filter tools based on Store preferences."""
  user_id = request.runtime.context.user_id

  # Read from Store: get user's enabled features
  store = request.runtime.store
  feature_flags = store.get(("features",), user_id)

  if feature_flags:
    enabled_features = feature_flags.value.get("enabled_tools", [])
    # Only include tools that are enabled for this user
    tools = [t for t in request.tools if t.name in enabled_features]
    request = request.override(tools=tools)

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "analysis_tool", "export_tool"],
  middleware=[store_based_tools],
  context_schema=Context,
  store=InMemoryStore()
)

# Filter tools based on user permissions from Runtime Context:
@dataclass
class Context:
  user_role: str

@wrap_model_call
def context_based_tools(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Filter tools based on Runtime Context permissions."""
  # Read from Runtime Context: get user role
  user_role = request.runtime.context.user_role

  if user_role == "admin":
    # Admins get all tools
    pass
  elif user_role == "editor":
    # Editors can't delete
    tools = [t for t in request.tools if t.name != "delete_data"]
    request = request.override(tools=tools)
  else:
    # Viewers get read-only tools
    tools = [t for t in request.tools if t.name.startswith("read_")]
    request = request.override(tools=tools)

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=["read_data", "write_data", "delete_data"],
  middleware=[context_based_tools],
  context_schema=Context
)

# --------------------------------------
# Model
# --------------------------------------

# Use different models based on conversation length from State:
from langchain.chat_models import init_chat_model

# Initialize models once outside the middleware
large_model = init_chat_model("claude-sonnet-4-5-20250929")
standard_model = init_chat_model("gpt-4o")
efficient_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def state_based_model(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Select model based on State conversation length."""
  # request.messages is a shortcut for request.state["messages"]

  message_count = len(request.messages)

  if message_count > 20:
    # Long conversation - use model with larger context window
    model = large_model
  elif message_count > 10:
    # Medium conversation
    model = standard_model
  else:
    # Short conversation - use efficient model
    model = efficient_model

  request = request.override(model=model)  

  return handler(request)

agent = create_agent(
  model="gpt-4o-mini",
  tools=[...],
  middleware=[state_based_model]
)

# Use user’s preferred model from Store:
@dataclass
class Context:
  user_id: str

# Initialize available models once
MODEL_MAP = {
  "gpt-4o": init_chat_model("gpt-4o"),
  "gpt-4o-mini": init_chat_model("gpt-4o-mini"),
  "claude-sonnet": init_chat_model("claude-sonnet-4-5-20250929"),
}

@wrap_model_call
def store_based_model(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Select model based on Store preferences."""
  user_id = request.runtime.context.user_id

  # Read from Store: get user's preferred model
  store = request.runtime.store
  user_preferences = store.get(("preferences",), user_id)

  if user_preferences:
    preferred_model = user_preferences.value.get("preferred_model")

    if preferred_model and preferred_model in MODEL_MAP:
      request = request.override(model=MODEL_MAP[preferred_model])

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[store_based_model],
  context_schema=Context,
  store=InMemoryStore()
)

# Select model based on cost limits or environment from Runtime Context:
@dataclass
class Context:
  cost_tier: str
  environment: str

# Initialize models once outside the middleware
premium_model = init_chat_model("claude-sonnet-4-5-20250929")
standard_model = init_chat_model("gpt-4o")
budget_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def context_based_model(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Select model based on Runtime Context."""
  # Read from Runtime Context: cost tier and environment

  cost_tier = request.runtime.context.cost_tier
  environment = request.runtime.context.environment

  if environment == "production" and cost_tier == "premium":
    # Production premium users get best model
    model = premium_model
  elif cost_tier == "budget":
    # Budget tier gets efficient model
    model = budget_model
  else:
    # Standard tier
    model = standard_model

  request = request.override(model=model)

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[context_based_model],
  context_schema=Context
)

# --------------------------------------
# Response Format
# --------------------------------------

# Defining formats
from pydantic import BaseModel, Field

class CustomerSupportTicket(BaseModel):
  """Structured ticket information extracted from customer message."""

  category: str = Field(description="Issue category: 'billing', 'technical', 'account', or 'product'")
  priority: str = Field(description="Urgency level: 'low', 'medium', 'high', or 'critical'")
  summary: str = Field(description="One-sentence summary of the customer's issue")
  customer_sentiment: str = Field(description="Customer's emotional tone: 'frustrated', 'neutral', or 'satisfied'")

# Selecting formats

# Configure structured output based on conversation state:
class SimpleResponse(BaseModel):
  """Simple response for early conversation."""
  answer: str = Field(description="A brief answer")

class DetailedResponse(BaseModel):
  """Detailed response for established conversation."""
  answer: str = Field(description="A detailed answer")
  reasoning: str = Field(description="Explanation of reasoning")
  confidence: float = Field(description="Confidence score 0-1")

@wrap_model_call
def state_based_output(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Select output format based on State."""
  # request.messages is a shortcut for request.state["messages"]

  message_count = len(request.messages)
  if message_count < 3:
    # Early conversation - use simple format
    request = request.override(response_format=SimpleResponse)  
  else:
    # Established conversation - use detailed format
    request = request.override(response_format=DetailedResponse)  

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[state_based_output]
)

# Configure output format based on user preferences in Store:

@dataclass
class Context:
  user_id: str

class VerboseResponse(BaseModel):
  """Verbose response with details."""
  answer: str = Field(description="Detailed answer")
  sources: list[str] = Field(description="Sources used")

class ConciseResponse(BaseModel):
  """Concise response."""
  answer: str = Field(description="Brief answer")

@wrap_model_call
def store_based_output(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Select output format based on Store preferences."""
  user_id = request.runtime.context.user_id

  # Read from Store: get user's preferred response style
  store = request.runtime.store
  user_preferences = store.get(("preferences",), user_id)

  if user_preferences:
    style = user_preferences.value.get("response_style", "concise")
    if style == "verbose":
      request = request.override(response_format=VerboseResponse)
    else:
      request = request.override(response_format=ConciseResponse)

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[store_based_output],
  context_schema=Context,
  store=InMemoryStore()
)

# Configure output format based on Runtime Context like user role or environment:

@dataclass
class Context:
  user_role: str
  environment: str

class AdminResponse(BaseModel):
  """Response with technical details for admins."""
  answer: str = Field(description="Answer")
  debug_info: dict = Field(description="Debug information")
  system_status: str = Field(description="System status")

class UserResponse(BaseModel):
  """Simple response for regular users."""
  answer: str = Field(description="Answer")

@wrap_model_call
def context_based_output(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Select output format based on Runtime Context."""
  # Read from Runtime Context: user role and environment
  user_role = request.runtime.context.user_role
  environment = request.runtime.context.environment

  if user_role == "admin" and environment == "production":
    # Admins in production get detailed output
    request = request.override(response_format=AdminResponse)
  else:
    # Regular users get simple output
    request = request.override(response_format=UserResponse)

  return handler(request)

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[context_based_output],
  context_schema=Context
)

# --------------------------------------
# Tool Context
# --------------------------------------

# Tools are special in that they both read and write context.
# Reads

# Read from State to check current session information:
from langchain.tools import ToolRuntime

@tool
def check_authentication(
  runtime: ToolRuntime
) -> str:
  """Check if user is authenticated."""
  # Read from State: check current auth status
  current_state = runtime.state
  is_authenticated = current_state.get("authenticated", False)

  if is_authenticated:
    return "User is authenticated"
  else:
    return "User is not authenticated"
  
agent = create_agent(
  model="gpt-4o",
  tools=[check_authentication]
)

# Read from Store to access persisted user preferences:

@dataclass
class Context:
  user_id: str

@tool
def get_preference(
  preference_key: str,
  runtime: ToolRuntime[Context]
) -> str:
  """Get user preference from Store."""
  user_id = runtime.context.user_id

  # Read from Store: get existing preferences
  store = runtime.store
  existing_preferences = store.get(("preferences",), user_id)

  if existing_preferences:
    value = existing_preferences.value.get(preference_key)
    return f"{preference_key}: {value}" if value else f"No preference set for {preference_key}"
  else:
    return "No preferences found"
  
agent = create_agent(
  model="gpt-4o",
  tools=[get_preference],
  context_schema=Context,
  store=InMemoryStore()
)

# Read from Runtime Context for configuration like API keys and user IDs:

@dataclass
class Context:
  user_id: str
  api_key: str
  db_connection: str

@tool
def fetch_user_data(
  query: str,
  runtime: ToolRuntime[Context]
) -> str:
  """Fetch data using Runtime Context configuration."""
  # Read from Runtime Context: get API key and DB connection
  user_id = runtime.context.user_id
  api_key = runtime.context.api_key
  db_connection = runtime.context.db_connection

  # Use configuration to fetch data
  results = perform_database_query(db_connection, query, api_key)

  return f"Found {len(results)} results for user {user_id}"

agent = create_agent(
  model="gpt-4o",
  tools=[fetch_user_data],
  context_schema=Context
)

# Invoke with runtime context
result = agent.invoke(
  {"messages": [{"role": "user", "content": "Get my data"}]},
  context=Context(
    user_id="user_123",
    api_key="sk-...",
    db_connection="postgresql://..."
  )
)

# Writes

# Write to State to track session-specific information using Command:
from langgraph.types import Command

@tool
def authenticate_user(
  password: str,
  runtime: ToolRuntime
) -> Command:
  """Authenticate user and update State."""
  # Perform authentication (simplified)
  if password == "correct":
    # Write to State: mark as authenticated using Command
    return Command(update={"authenticated": True})
  else:
    return Command(update={"authenticated": False})

agent = create_agent(
  model="gpt-4o",
  tools=[authenticate_user]
)

# Write to Store to persist data across sessions:

@dataclass
class Context:
  user_id: str

@tool
def save_preference(
  preference_key: str,
  preference_value: str,
  runtime: ToolRuntime[Context]
) -> str:
  """Save user preference to Store."""
  user_id = runtime.context.user_id

  # Read existing preferences
  store = runtime.store
  existing_preferences = store.get(("preferences",), user_id)

  # Merge with new preference
  preferences = existing_preferences.value if existing_preferences else {}
  preferences[preference_key] = preference_value

  # Write to Store: save updated preferences
  store.put(("preferences",), user_id, preferences)

  return f"Saved preference: {preference_key} = {preference_value}"

agent = create_agent(
  model="gpt-4o",
  tools=[save_preference],
  context_schema=Context,
  store=InMemoryStore()
)

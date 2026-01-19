# ===========================================================================
# Built-in middleware
# ===========================================================================

# Provider-agnostic middleware

# 1. Summarization
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# Single condition: trigger if tokens >= 4000
agent = create_agent(
  model="gpt-4o",
  tools=["weather_tool", "calculator_tool"],
  middleware=[
    SummarizationMiddleware(
      model="gpt-4o-mini",
      trigger=("tokens", 4000),
      keep=("messages", 20)
    )
  ]
)

# Multiple conditions: trigger if number of tokens >= 3000 OR messages >= 6
agent_2 = create_agent(
  model="gpt-5",
  tools=["weather_tool", "calculator_tool"],
  middleware=[
    SummarizationMiddleware(
      model="gpt-4o-mini",
      trigger=[
        ("tokens", 3000),
        ("messages", 6)
      ],
      keep=("messages", 20)
    )
  ]
)

# Using fractional limits
agent_3 = create_agent(
  model="gpt-5",
  tools=["weather_tool", "calculator_tool"],
  middleware=[
    SummarizationMiddleware(
      model="gpt-4o-mini",
      trigger=("fraction", 0.8),
      keep=("fraction", 0.3)
    )
  ]
)

# 2. HumanInTheLoopMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def read_email_tool(email_id: str) -> str:
  """Mock function to read an email by its ID."""
  return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
  """Mock function to send an email."""
  return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
  model="gpt-4o",
  tools=[read_email_tool, send_email_tool],
  checkpointer=InMemorySaver(),
  middleware=[
    HumanInTheLoopMiddleware(
      interrupt_on={
        "read_email_tool": False,
        "send_email_tool": {
          "allowed_decisions": ["approve", "edit", "reject"],
        },
      }
    )
  ]
)

# 3. ModelCallLimitMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
  model="gpt-4o",
  checkpointer=InMemorySaver(),
  tools=[],
  middleware=[
    ModelCallLimitMiddleware(
      thread_limit=10,
      run_limit=5,
      exit_behavior="end",
    )
  ]
)

# 4. ToolCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "database_tool"],
  middleware=[
    # Global limit
    ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
    # Tool-specific limit
    ToolCallLimitMiddleware(
      tool_name="search",
      thread_limit=5,
      run_limit=3
    )
  ]
)

global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
search_limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3)
database_limiter = ToolCallLimitMiddleware(tool_name="query_database", thread_limit=10)
strict_limiter = ToolCallLimitMiddleware(tool_name="scrape_webpage", run_limit=2, exit_behavior="error")

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "database_tool", "scraper_tool"],
  middleware=[global_limiter, search_limiter, database_limiter, strict_limiter],
)

# 5. ModelFallbackMiddleware
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[
    ModelFallbackMiddleware(
      "gpt-4o-mini",
      "claude-3-5-sonnet-20241022"
    )
  ]
)

# 6. PIIMiddleware
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    PIIMiddleware("credit_card", strategy="mask", apply_to_input=True)
  ]
)

# Customer PII Types
import re

# Method 1: Regex pattern string
agent1 = create_agent(
  model="gpt-5",
  tools=[],
  middleware=[
    PIIMiddleware(
      "api_key",
      detector=r"sk-[a-zA-Z0-9]{32}",
      strategy="block"
    )
  ]
)

# Method 2: Compiled regex pattern
agent2 = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[
    PIIMiddleware(
      "phone_number",
      detector=re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}"),
      strategy="mask"
    )
  ]
)

# Method 3: Custom detector function
def detect_ssn(content: str) -> list[dict[str, str | int]]:
  """Detect SSN with validation.

  Returns a list of dictionaries with 'text', 'start', and 'end' keys.
  """

  matches = []
  pattern = r"\d{3}-\d{2}-\d{4}"
  for match in re.finditer(pattern, content):
    ssn = match.group(0)
    # Validate: first 3 digits shouldn't be 000, 666, or 900-999
    first_three = int(ssn[:3])
    if first_three not in [0, 666] and not (900 <= first_three <= 999):
      matches.append({
        "text": ssn,
        "start": match.start(),
        "end": match.end(),
      })
  return matches

agent3 = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[
    PIIMiddleware(
      "ssh",
      detector=detect_ssn,
      strategy="hash"
    )
  ]
)

# 7. TodoListMiddleware
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=["read_file", "write_file", "run_tests"],
  middleware=[TodoListMiddleware()]
)

# 8. LLMToolSelectorMiddleware
from langchain.agents.middleware import LLMToolSelectorMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=["tool1", "tool2", "tool3", "tool4", "tool5", ...],
  middleware=[
    LLMToolSelectorMiddleware(
      model="gpt-4o-mini",
      max_tools=3,
      always_include=["search"]
    )
  ]
)

# 9. ToolRetryMiddleware
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "database_tool"],
  middleware=[
    ToolRetryMiddleware(
      max_retries=3,
      backoff_factor=2.0,
      initial_delay=1.0,
      max_delay=60.0,
      jitter=True,
      tools=["api_tool"],
      retry_on=(ConnectionError, TimeoutError),
      on_failure="continue"
    )
  ]
)

# 10. ModelRetryMiddleware
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "database_tool"],
  middleware=[
    ModelRetryMiddleware(
      max_retries=3,
      backoff_factor=2.0,
      initial_delay=1.0,
    )
  ]
)

# Full example

# Basic usage with default settings (2 retries, exponential backoff)
agent = create_agent(
  model="gpt-4o",
  tools=["search_tool"],
  middleware=[ModelRetryMiddleware()],
)

# Custom exception filtering
class TimeoutError(Exception):
  """Custom exception for timeout errors."""
  pass

class ConnectionError(Exception):
  """Custom exception for connection errors."""
  pass

# Retry specific exceptions only
retry = ModelRetryMiddleware(
  max_retries=4,
  retry_on=(TimeoutError, ConnectionError),
  backoff_factor=1.5,
)

def should_retry(error: Exception) -> bool:
  # Only retry on rate limit errors
  if isinstance(error, TimeoutError):
      return True
  # Or check for specific HTTP status codes
  if hasattr(error, "status_code"):
      return error.status_code in (429, 503)
  return False

# Return error message instead of raising
retry_with_filter = ModelRetryMiddleware(
  max_retries=3,
  retry_on=should_retry,
)

# Custom error message formatting
def format_error(error: Exception) -> str:
  return f"Model call failed: {error}. Please try again later."

retry_with_formatter = ModelRetryMiddleware(
  max_retries=4,
  on_failure=format_error,
)

# Constant backoff (no exponential growth)
constant_backoff = ModelRetryMiddleware(
  max_retries=5,
  backoff_factor=0.0,  # No exponential growth
  initial_delay=2.0,  # Always wait 2 seconds
)

# Raise exception on failure
strict_retry = ModelRetryMiddleware(
  max_retries=2,
  on_failure="error",  # Re-raise exception instead of returning message
)

# 11. LLMToolEmulator
from langchain.agents.middleware import LLMToolEmulator

agent = create_agent(
  model="gpt-4o",
  tools=["get_weather", "search_database", "send_email"],
  middleware=[
    LLMToolEmulator(), # Emulate all tools
  ],
)

# Full example
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
  """Get the current weather for a location."""
  return f"Weather in {location}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
  """Send an email."""
  return "Email sent"

# Emulate all tools (default behavior)
agent = create_agent(
  model="gpt-4o",
  tools=[get_weather, send_email],
  middleware=[LLMToolEmulator()],
)

# Emulate specific tools only
agent2 = create_agent(
  model="gpt-4o",
  tools=[get_weather, send_email],
  middleware=[LLMToolEmulator(tools=["get_weather"])],
)

# Use custom model for emulation
agent4 = create_agent(
  model="gpt-4o",
  tools=[get_weather, send_email],
  middleware=[LLMToolEmulator(model="claude-sonnet-4-5-20250929")],
)

# 12. ContextEditingMiddleware
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "your_calculator_tool", "database_tool"],
  middleware=[
    ContextEditingMiddleware(
      edits=[
        ClearToolUsesEdit(
          trigger=10000,
          keep=3,
          clear_tool_inputs=False,
          exclude_tools=[],
          placeholder="[cleared]",
        )
      ]
    )
  ]
)

# 13. ShellToolMiddleware
from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy, DockerExecutionPolicy, RedactionRule
from langchain.messages import HumanMessage

# Basic shell tool with host execution
agent = create_agent(
  model="gpt-4o",
  tools=["search_tool"],
  middleware=[
    ShellToolMiddleware(
      workspace_root="/workspace",
      execution_policy=HostExecutionPolicy()
    )
  ]
)

# Docker isolation with startup commands
agent_docker = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[
    ShellToolMiddleware(
      workspace_root="/workspace",
      startup_commands=["pip install requests", "export PYTHONPATH=/workspace"],
      execution_policy=DockerExecutionPolicy(
        image="python:3.11-slim",
        command_timeout=60.0,
      ),
    ),
  ],
)

# With output redaction (applied post execution)
agent_redacted = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[
    ShellToolMiddleware(
      workspace_root="/workspace",
      redaction_rules=[
        RedactionRule(pii_type="api_key", detector=r"sk-[a-zA-Z0-9]{32}"),
      ],
    ),
  ],
)

# 14. FilesystemFileSearchMiddleware
from langchain.agents.middleware import FilesystemFileSearchMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=[],
  middleware=[
    FilesystemFileSearchMiddleware(
      root_path="/workspace",
      use_ripgrep=True,
      max_file_size_mb=10
    )
  ]
)

# Agent can now use glob_search and grep_search tools
result = agent.invoke({
  "messages": [HumanMessage("Find all Python files containing 'async def'")]
})

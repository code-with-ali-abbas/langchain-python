# ===========================================================================
# Model Context Protocol (MCP)
# ===========================================================================

# Accessing multiple MCP servers
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
  client = MultiServerMCPClient(
    {
      "math": {
        "transport": "stdio",  # Local subprocess communication
        "command": "python",
        # Absolute path to your math_server.py file
        "args": ["/path/to/math_server.py"],
      },
      "weather": {
        "transport": "http",  # HTTP-based remote server
        # Ensure you start your weather server on port 8000
        "url": "http://localhost:8000/mcp",
      }
    }
  )

  tools = await client.get_tools()

  agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=tools
  )

  math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
  )

  weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
  )

# ---------------------------------------------------
# Custom servers
# ---------------------------------------------------

# To test your agent with MCP tool servers, use the following examples:
from fastmcp import FastMCP

mcp = FastMCP("math")

@mcp.tool()
def add(a: int, b: int) -> int:
  """Add two numbers"""
  return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
  """Multiply two numbers"""
  return a * b

if __name__ == "__main__":
  mcp.run(transport="stdio")

# ---------------------------------------------------
# Transports
# ---------------------------------------------------

# HTTP

client = MultiServerMCPClient(
  {
    "weather": {
      "transport": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {                                       # Passing headers
        "Authorization": "Bearer YOUR_TOKEN",
        "X-Custom-Header": "custom-value"
      },
      "auth": "auth",                                    # Authentication
    }
  }
)

# stdio
client = MultiServerMCPClient(
  {
    "math": {
      "transport": "stdio",
      "command": "python",
      "args": ["/path/to/math_server.py"],
    }
  }
)

# ---------------------------------------------------
# Stateful Sessions
# ---------------------------------------------------

def load_mcp_tools(session):
  return session.get_tools()

async def session_example():
  async with client.session("server_name") as session:
    tools = await load_mcp_tools(session)

    agent = create_agent(
      model="anthropic:claude-3-7-sonnet-latest",
      tools=tools
    )

# ---------------------------------------------------
# Core Features
# ---------------------------------------------------

# Tools
from langchain.messages import ToolMessage

async def tools_example():
  # Loading tools
  client = MultiServerMCPClient({...})
  tools = await client.get_tools()
  agent = create_agent(model="claude-sonnet-4-5-20250929", tools=tools)

  # Structured content
  result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Get data from the server"}]}
  )

  # Extract structured content from tool messages
  for message in result["messages"]:
    if isinstance(message, ToolMessage) and message.artifact:
      structured_content = message.artifact["structured_content"]

# Appending structured content via interceptor
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import TextContent

async def append_structured_content(request: MCPToolCallRequest, handler):
  """Append structured content from artifact to tool message."""

  result = await handler(request)
  if result.structuredContent:
    result.content += [
      TextContent(type="text", text=json.dumps(result.structuredContent)),
    ]

  return result

client = MultiServerMCPClient({...}, tool_interceptors=append_structured_content)

# Multimodal tool content

client = MultiServerMCPClient({...})

async def multimodal_example():
  tools = await client.get_tools()
  agent = create_agent(model="claude-sonnet-4-5-20250929", tools=tools)

  result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Take a screenshot of the current page"}]}
  )

  # Access multimodal content from tool messages
  for message in result["messages"]:
    if message.type == "tool":
      # Raw content in provider-native format
      print(f"Raw content: {message.content}")

      # Standardized content blocks #
      for block in message.content_blocks:
        if block["type"] == "text":
          print(f"Text: {block['text']}")
        elif block["type"] == "image":
          print(f"Image URL: {block.get('url')}")
          print(f"Image base64: {block.get('base64', '')[:50]}...")

# ---------------------------------------------------
# Resources
# ---------------------------------------------------

# Loading resources
async def resources_example():
  client = MultiServerMCPClient({...})

  # Load all resources from a server
  blobs = await client.get_resources("server_name")

  # Or load specific resources by URI
  blobs = await client.get_resources("server_name", uris=["file:///path/to/file.txt"])

  for blob in blobs:
    print(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
    print(blob.as_string())  # For text content

  # You can also use load_mcp_resources directly with a session for more control:
  async with client.session("server_name") as session:
    # Load all resources
    blobs = await load_mcp_resources(session)

    # Or load specific resources by URI
    blobs = await load_mcp_resources(session, uris=["file:///path/to/file.txt"])

# ---------------------------------------------------
# Prompts
# ---------------------------------------------------
from langchain_mcp_adapters.prompts import load_mcp_prompt

async def prompts_example():
  client = MultiServerMCPClient({...})

  # Load a prompt by name
  messages = await client.get_prompt("server_name", "summarize")

  # Load a prompt with arguments
  messages = await client.get_prompt(  
    "server_name",  
    "code_review",  
    arguments={"language": "python", "focus": "security"}  
  )

  # Use the messages in your workflow
  for message in messages:
    print(f"{message.type}: {message.content}")



  async with client.session("server_name") as session:
    # Load a prompt by name
    messages = await load_mcp_prompt(session, "summarize")

    # Load a prompt with arguments
    messages = await load_mcp_prompt(
      session,
      "code_review",
      arguments={"language": "python", "focus": "security"}
    )

# ===========================================================================
# Advanced Features
# ===========================================================================

# ---------------------------------------------------
# Tool interceptors
# ---------------------------------------------------

# Accessing runtime context

# Inject user context into MCP tool calls
from dataclasses import dataclass

@dataclass
class Context:
  user_id: str
  api_key: str

async def inject_user_context(
  request: MCPToolCallRequest,
  handler,
):
  """Inject user credentials into MCP tool calls."""
  runtime = request.runtime
  user_id = runtime.context.user_id  
  api_key = runtime.context.api_key  

  # Add user context to tool arguments
  modified_request = request.override(
    args={**request.args, "user_id": user_id}
  )
  return await handler(modified_request)

client = MultiServerMCPClient(
  {...},
  tool_interceptors=[inject_user_context],
)

async def user_context_example():
  tools = await client.get_tools()
  agent = create_agent("gpt-4o", tools, context_schema=Context)

  # Invoke with user context
  result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Search my orders"}]},
    context={"user_id": "user_123", "api_key": "sk-..."}
  )

# Access user preferences from store
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
  user_id: str

async def personalize_search(
  request: MCPToolCallRequest,
  handler,
):
  """Personalize MCP tool calls using stored preferences."""
  runtime = request.runtime
  user_id = runtime.context.user_id
  store = runtime.store

  # Read user preferences from store
  preferences = store.get(("preferences",), user_id)

  if preferences and request.name == "search":
    # Apply user's preferred language and result limit
    modified_args = {
      **request.args,
      "language": preferences.value.get("language", "en"),
      "limit": preferences.value.get("result_limit", 10),
    }
    request = request.override(args=modified_args)

  return await handler(request)

client = MultiServerMCPClient(
  {...},
  tool_interceptors=[personalize_search],
)

async def user_preferences_example():
  tools = await client.get_tools()
  agent = create_agent(
    "gpt-4o",
    tools,
    context_schema=Context,
    store=InMemoryStore()
  )

# Filter tools based on authentication state
async def require_authentication(
  request: MCPToolCallRequest,
  handler,
):
  """Block sensitive MCP tools if user is not authenticated."""
  runtime = request.runtime
  state = runtime.state
  is_authenticated = state.get("authenticated", False)

  sensitive_tools = ["delete_file", "update_settings", "export_data"]

  if request.name in sensitive_tools and not is_authenticated:
    # Return error instead of calling tool
    return ToolMessage(
      content="Authentication required. Please log in first.",
      tool_call_id=runtime.tool_call_id,
    )
  
  return await handler(request)

client = MultiServerMCPClient(
  {...},
  tool_interceptors=[require_authentication],
)

# Return custom responses with tool call ID

async def rate_limit_interceptor(
    request: MCPToolCallRequest,
    handler,
):
  """Rate limit expensive MCP tool calls."""
  runtime = request.runtime
  tool_call_id = runtime.tool_call_id

  # Check rate limit (simplified example)
  if is_rate_limited(request.name):
    return ToolMessage(
      content="Rate limit exceeded. Please try again later.",
      tool_call_id=tool_call_id,  
    )
  
  result = await handler(request)

  # Log successful tool call
  log_tool_execution(tool_call_id, request.name, success=True)

  return result

client = MultiServerMCPClient(
  {...},
  tool_interceptors=[rate_limit_interceptor],
)

# State updates and commands
from langgraph.types import Command

async def handle_task_completion(
  request: MCPToolCallRequest,
  handler,
):
  """Mark task complete and hand off to summary agent."""
  result = await handler(request)

  if request.name == "submit_order":
    return Command(
      update={
        "messages": [result] if isinstance(result, ToolMessage) else [],
        "task_status": "completed",  
      },
      goto="summary_agent",  
    )
  
  return result

# Use Command with goto="__end__" to end execution early:

async def end_on_success(
  request: MCPToolCallRequest,
  handler,
):
  """End agent run when task is marked complete."""
  result = await handler(request)

  if request.name == "mark_complete":
    return Command(
      update={"messages": [result], "status": "done"},
      goto="__end__",  
    )

  return result

# ---------------------------------------------------
# Custom interceptors
# ---------------------------------------------------

# Basic pattern

async def logging_interceptor(
  request: MCPToolCallRequest,
  handler,
):
  """Log tool calls before and after execution."""
  print(f"Calling tool: {request.name} with args: {request.args}")
  result = await handler(request)
  print(f"Tool {request.name} returned: {result}")

  return result

client = MultiServerMCPClient(
  {"math": {"transport": "stdio", "command": "python", "args": ["/path/to/server.py"]}},
  tool_interceptors=[logging_interceptor],  
)

# Modifying requests
# Modifying tool arguments

async def double_args_interceptor(
  request: MCPToolCallRequest,
  handler,
):
  """Double all numeric arguments before execution."""
  modified_args = {k: v * 2 for k, v in request.args.items()}
  modified_request = request.override(args=modified_args)

  return await handler(modified_request)

  # Original call: add(a=2, b=3) becomes add(a=4, b=6)

# Modifying headers at runtime

async def auth_header_interceptor(
  request: MCPToolCallRequest,
  handler,
):
  """Add authentication headers based on the tool being called."""
  token = get_token_for_tool(request.name)
  modified_request = request.override(
    headers={"Authorization": f"Bearer {token}"}  
  )

  return await handler(modified_request)

# Composing interceptors

async def outer_interceptor(request, handler):
  print("outer: before")
  result = await handler(request)
  print("outer: after")
  return result

async def inner_interceptor(request, handler):
  print("inner: before")
  result = await handler(request)
  print("inner: after")
  return result

client = MultiServerMCPClient(
  {...},
  tool_interceptors=[outer_interceptor, inner_interceptor],  
)

# Execution order:
# outer: before -> inner: before -> tool execution -> inner: after -> outer: after

# Error handling
# Retry on error

import asyncio

async def retry_interceptor(
  request: MCPToolCallRequest,
  handler,
  max_retries: int = 3,
  delay: float = 1.0,
):
  """Retry failed tool calls with exponential backoff."""
  last_error = None

  for attempt in range(max_retries):
    try:
      return await handler(request)
    except Exception as e:
      last_error = e
      if attempt < max_retries - 1:
        wait_time = delay * (2 ** attempt) # Exponential backoff
        print(f"Tool {request.name} failed (attempt {attempt + 1}), retrying in {wait_time}s...")
        await asyncio.sleep(wait_time)

  raise last_error

client = MultiServerMCPClient(
  {...},
  tool_interceptors=[retry_interceptor],  
)

# Error handling with fallback

async def fallback_interceptor(
  request: MCPToolCallRequest,
  handler,
):
  """Return a fallback value if tool execution fails."""
  try:
    return await handler(request)
  except TimeoutError:
    return f"Tool {request.name} timed out. Please try again later."
  except ConnectionError:
    return f"Could not connect to {request.name} service. Using cached data."
  
# ---------------------------------------------------
# Progress Notifications
# ---------------------------------------------------

from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

async def on_progress(
  progress: float,
  total: float | None,
  message: str | None,
  context: CallbackContext,
):
  """Handle progress updates from MCP servers."""
  percent = (progress / total * 100) if total else progress
  tool_info = f" ({context.tool_name})" if context.tool_name else ""
  print(f"[{context.server_name}{tool_info}] Progress: {percent:.1f}% - {message}")

client = MultiServerMCPClient(
  {...},
  callbacks=Callbacks(on_progress=on_progress),  
)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------

from mcp.types import LoggingMessageNotificationParams

async def on_logging_message(
  params: LoggingMessageNotificationParams,
  context: CallbackContext,
):
  """Handle log messages from MCP servers."""
  print(f"[{context.server_name}] {params.level}: {params.data}")

client = MultiServerMCPClient(
  {...},
  callbacks=Callbacks(on_logging_message=on_logging_message),  
)

# ---------------------------------------------------
# Elicitation
# ---------------------------------------------------

# Server setup

from pydantic import BaseModel
from mcp.server.fastmcp import Context

server = FastMCP("Profile")

class UserDetails(BaseModel):
  email: str
  age: int

@server.tool()
async def create_profile(name: str, ctx: Context) -> str:
  """Create a user profile, requesting details via elicitation."""

  result = await ctx.elicit(  
    message=f"Please provide details for {name}'s profile:",  
    schema=UserDetails,  
  )

  if result.action == "accept" and result.data:
    return f"Created profile for {name}: email={result.data.email}, age={result.data.age}"
  if result.action == "decline":
    return f"User declined. Created minimal profile for {name}."
  return "Profile creation cancelled."

if __name__ == "__main__":
  server.run(transport="http")

# Client setup

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

async def on_elicitation(
  mcp_context: RequestContext,
  params: ElicitRequestParams,
  context: CallbackContext,
) -> ElicitResult:
  """Handle elicitation requests from MCP servers."""
  # In a real application, you would prompt the user for input
  # based on params.message and params.requestedSchema

  return ElicitResult(  
    action="accept",  
    content={"email": "user@example.com", "age": 25},  
  )

client = MultiServerMCPClient(
  {
    "profile": {
      "url": "http://localhost:8000/mcp",
      "transport": "http",
    }
  },
  callbacks=Callbacks(on_elicitation=on_elicitation),  
)

# Response actions

# Accept with data
ElicitResult(action="accept", content={"email": "user@example.com", "age": 25})

# Decline (user doesn't want to provide info)
ElicitResult(action="decline")

# Cancel (abort the operation)
ElicitResult(action="cancel")

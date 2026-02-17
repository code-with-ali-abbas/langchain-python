# ===========================================================================
# Subagents
# ===========================================================================

# -------------------------------------------
# Basic Implementation
# -------------------------------------------

from langchain.tools import tool
from langchain.agents import create_agent

# Create a subagent
subagent = create_agent(model="anthropic:claude-sonnet-4-20250514", tools=[...])

# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
  result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
  return result["messages"][-1].content

# Main agent with subagent as a tool
main_agent = create_agent(model="anthropic:claude-sonnet-4-20250514", tools=[call_research_agent])

# ===========================================================================
# Tool Patterns
# ===========================================================================

# -------------------------------------------
# Tool Per Agent
# -------------------------------------------

# Create a sub-agent
subagent = create_agent(model="...", tools=[...])

# Wrap it as a tool
@tool("subagent_name", description="subagent_description")
def call_subagent(query: str):
  result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
  return result["messages"][-1].content

# Main agent with subagent as a tool
main_agent = create_agent(model="...", tools=[call_subagent])

# -------------------------------------------
# Single Dispatch Tool
# -------------------------------------------

# Sub-agents developed by different teams
research_agent = create_agent(
  model="gpt-4.1",
  prompt="You are a research specialist..."
)

writer_agent = create_agent(
  model="gpt-4.1",
  prompt="You are a writing specialist..."
)

# Registry of available sub-agents
SUBAGENTS = {
  "research": research_agent,
  "writer": writer_agent,
}

@tool
def task(agent_name: str, description: str) -> str:
  """Launch an ephemeral subagent for a task.

  Available agents:
  - research: Research and fact-finding
  - writer: Content creation and editing
  """

  agent = SUBAGENTS[agent_name]

  result = agent.invoke({
    "messages": [{"role": "user", "content": description}]
  })

  return result["messages"][-1].content

# Main coordinator agent
main_agent = create_agent(
  model="gpt-4.1",
  tools=[task],
  system_prompt=(
    "You coordinate specialized sub-agents. "
    "Available: research (fact-finding), "
    "writer (content creation). "
    "Use the task tool to delegate work."
  ),
)

# ===========================================================================
# Context engineering
# ===========================================================================

# -------------------------------------------
# Subagents Specs
# -------------------------------------------

# System prompt enumeration
main_agent = create_agent(
  model="...",
  tools=[task],
  system_prompt=(
    "You coordinate specialized sub-agents. "
    "Available agents:\n"
    "- research: Research and fact-finding\n"
    "- writer: Content creation and editing\n"
    "- reviewer: Code and document review\n"
    "Use the task tool to delegate work."
  ),
)

# Enum constraint on dispatch tool
from enum import Enum

class AgentName(str, Enum):
  RESEARCH = "research"
  WRITER = "writer"
  REVIEWER = "reviewer"

@tool
def task(
  agent_name: AgentName,  # Enum constraint
  description: str
) -> str:
  """Launch an ephemeral subagent for a task."""

# Tool-based discovery
def search_agent_registry(query: str) -> list:
    raise NotImplementedError

def format_agent_list(agents):
    raise NotImplementedError
  # ...

@tool
def list_agents(query: str = "") -> str:
  """List available subagents, optionally filtered by query."""
  agents = search_agent_registry(query)
  return format_agent_list(agents)

@tool
def task(agent_name: str, description: str) -> str:
  """Launch an ephemeral subagent for a task."""
  # ...

main_agent = create_agent(
  model="...",
  tools=[task, list_agents],
  system_prompt="Use list_agents to discover available subagents, then use task to invoke them."
)

# -------------------------------------------
# Subagents inputs
# -------------------------------------------

from langchain.agents import AgentState
from langchain.tools import ToolRuntime

subagent1 = create_agent(model="anthropic:claude-sonnet-4-20250514", tools=[...])

class CustomState(AgentState):
  example_state_key: str

def some_logic(query, messages):
    raise NotImplementedError

@tool("subagent1_name", description="subagent1_description")
def call_subagent1(query: str, runtime: ToolRuntime[None, CustomState]):
  # Apply any logic needed to transform the messages into a suitable input
  subagent_input = some_logic(query, runtime.state["messages"])
  result = subagent1.invoke({
    "messages": subagent_input,
    # You could also pass other state keys here as needed.
    # Make sure to define these in both the main and subagent's
    # state schemas.
    "example_state_key": runtime.state["example_state_key"]
  })

  return result["messages"][-1].content

# -------------------------------------------
# Subagents outputs
# -------------------------------------------

from typing import Annotated
from langchain.tools import InjectedToolCallId
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool("subagent1_name", description="subagent1_description")
def call_subagent1(
  query: str,
  tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
  result = subagent1.invoke({
    "messages": [{"role": "user", "content": query}]
  })
  return Command(update={
    # Pass back additional state from the subagent
    "example_state_key": result["example_state_key"],
    "messages": [
      ToolMessage(
        content=result["messages"][-1].content,
        tool_call_id=tool_call_id
      )
    ]
  })

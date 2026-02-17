# ===========================================================================
# Router
# ===========================================================================

# ---------------------------------
# Basic implementation
# ---------------------------------

# Single Agent
# Use Command to route to a single specialized agent

from langgraph.types import Command

def classify_query(query: str) -> str:
  """Use LLM to classify query and determine the appropriate agent."""
  # Classification logic here
  ...

def route_query(state: State) -> Command:
  """Route to the appropriate agent based on query classification."""
  active_agent = classify_query(state["query"])

  # Route to the selected agent
  return Command(goto=active_agent)


# Multi Agents (parallel)
# Use Send to fan out to multiple specialized agents in parallel
from typing import TypedDict
from langgraph.types import Send

class ClassificationResult(TypedDict):
  query: str
  agent: str

def classify_query(query: str) -> list[ClassificationResult]:
  """Use LLM to classify query and determine which agents to invoke."""
  # Classification logic here
  ...

def route_query(state: State):
  """Route to relevant agents based on query classification."""
  classifications = classify_query(state["query"])

  # Fan out to selected agents in parallel
  return [
    Send(c["agent"], {"query": c["query"]})
    for c in classifications
  ]

# ---------------------------------
# Two approaches
# ---------------------------------

# 1. Stateless routers
# **********************************************

# 2. Stateful routers
# Tool wrapper

@tool
def search_docs(query: str) -> str:
  """Search across multiple documentation sources."""
  result = workflow.invoke({"query": query})  
  return result["final_answer"]

# Conversational agent uses the router as a tool
conversational_agent = create_agent(
  model,
  tools=[search_docs],
  prompt="You are a helpful assistant. Use search_docs to answer questions."
)

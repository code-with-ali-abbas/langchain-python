# ===========================================================================
# Overview
# ===========================================================================

# - Control and customize agent execution at every step
# - Tracking agent behavior with logging, analytics and debugging
# - Transforming prompts, tool selection, and output formatting
# - Adding retries, fallbacks, and early termination logic
# - Applying rate limits, guardrails, and PII detection

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=[...],
  middleware=[
    SummarizationMiddleware(...),
    HumanInTheLoopMiddleware(...)
  ]
)

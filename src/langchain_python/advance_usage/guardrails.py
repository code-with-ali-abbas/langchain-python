# ===========================================================================
# Guardrails
# ===========================================================================

# -----------------------------------------------
# Built-in guardrails
# -----------------------------------------------

# 1. PII Detection
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
  model="gpt-4o",
  tools=["customer_service_tool", "email_tool"],
  middleware=[
    PIIMiddleware(
      "email",
      strategy="redact",
      apply_to_input=True
    ),
    PIIMiddleware(
      "credit_card",
      strategy="mask",
      apply_to_input=True
    ),
    PIIMiddleware(
      "api_key",
      detector=r"sk-[a-zA-Z0-9]{32}",
      strategy="block",
      apply_to_input=True
    )
  ]
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100"}]
})

# 2. Human-in-the-loop
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "send_email_tool", "delete_database_tool"],
  middleware=[
      HumanInTheLoopMiddleware(
          interrupt_on={
              "send_email": True,
              "delete_database": True,
              "search": False,
          }
      ),
  ],
  # Persist the state across interrupts
  checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "some_id"}}

result = agent.invoke(
  {"messages": [{"role": "user", "content": "Send an email to the team"}]},
  config=config
)

result = agent.invoke(
  Command(resume={"decisions": [{"type": "approve"}]}),
  config=config
)

# -----------------------------------------------
# Custom guardrails
# -----------------------------------------------

# Before agent guardrails

# class syntax
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class ContentFilterMiddleware(AgentMiddleware):
  """Deterministic guardrail: Block requests containing banned keywords."""

  def __init__(self, banned_keywords: list[str]):
    super().__init__()
    self.banned_keywords = [kw.lower() for kw in banned_keywords]

  @hook_config(can_jump_to=["end"])
  def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if not state["messages"]:
      return None

    first_message = state["messages"][0]
    if first_message.type != "human":
      return None

    content = first_message.content.lower()

    for keyword in self.banned_keywords:
      if keyword in content:
        return {
          "messages": [{
            "role": "assistant",
            "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
          }],
          "jump_to": "end"
        }

    return None
  
agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "calculator_tool"],
  middleware=[
    ContentFilterMiddleware(
      banned_keywords=["hack", "exploit", "malware"]
    ),
  ],
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})

# decorator syntax
from langchain.agents.middleware import before_agent

banned_keywords = ["hack", "exploit", "malware"]

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  """Deterministic guardrail: Block requests containing banned keywords."""
  if not state["messages"]:
    return None

  first_message = state["messages"][0]
  if first_message.type != "human":
    return None

  content = first_message.content.lower()

  for keyword in banned_keywords:
    if keyword in content:
      return {
        "messages": [{
          "role": "assistant",
          "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
        }],
        "jump_to": "end"
      }

  return None

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "calculator_tool"],
  middleware=[content_filter],
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})

# After agent guardrails

# class syntax
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model

class SafetyGuardrailMiddleware(AgentMiddleware):
  """Model-based guardrail: Use an LLM to evaluate response safety."""

  def __init__(self):
    super().__init__()
    self.safety_model = init_chat_model("gpt-4o-mini")

  @hook_config(can_jump_to=["end"])
  def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if not state["messages"]:
      return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
      return None

    safety_prompt = f"""Evaluate if this response is safe and appropriate.
    Respond with only 'SAFE' or 'UNSAFE'.

    Response: {last_message.content}"""

    result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

    if "UNSAFE" in result.content:
      last_message.content = "I cannot provide that response. Please rephrase your request."

    return None
  
agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "calculator_tool"],
  middleware=[SafetyGuardrailMiddleware()],
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "How do I make explosives?"}]
})

# decorator syntax
from langchain.agents.middleware import after_agent

safety_model = init_chat_model("gpt-4o-mini")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  """Model-based guardrail: Use an LLM to evaluate response safety."""
  if not state["messages"]:
    return None

  last_message = state["messages"][-1]
  if not isinstance(last_message, AIMessage):
    return None

  safety_prompt = f"""Evaluate if this response is safe and appropriate.
  Respond with only 'SAFE' or 'UNSAFE'.

  Response: {last_message.content}"""

  result = safety_model.invoke([{"role": "user", "content": safety_prompt}])

  if "UNSAFE" in result.content:
    last_message.content = "I cannot provide that response. Please rephrase your request."

  return None

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "calculator_tool"],
  middleware=[safety_guardrail],
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "How do I make explosives?"}]
})

# Combine multiple guardrails

agent = create_agent(
  model="gpt-4o",
  tools=["search_tool", "send_email_tool"],
  middleware=[
    # Layer 1: Deterministic input filter (before agent)
    ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

    # Layer 2: PII protection (before and after model)
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    PIIMiddleware("email", strategy="redact", apply_to_output=True),

    # Layer 3: Human approval for sensitive tools
    HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

    # Layer 4: Model-based safety check (after agent)
    SafetyGuardrailMiddleware(),
  ],
)

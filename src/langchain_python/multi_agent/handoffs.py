# ===========================================================================
# Handoffs
# ===========================================================================

# ------------------------------------
# Basic implementation
# ------------------------------------

from langchain.tools import tool
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def transfer_to_specialist(runtime) -> Command:
  """Transfer to the specialist agent."""
  return Command(
    update={
      "messages": [
        ToolMessage(  
          content="Transferred to specialist",
          tool_call_id=runtime.tool_call_id  
        )
      ],
      "current_step": "specialist"  # Triggers behavior change
    }
  )

# ===========================================================================
# Implementation approaches
# ===========================================================================

# ------------------------------------
# Single agent with middleware
# ------------------------------------

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from typing import Callable

# 1. Define state with current_step tracker
class SupportState(AgentState):
  """Track which step is currently active."""
  current_step: str = "triage"
  warranty_status: str | None = None

# 2. Tools update current_step via Command
@tool
def record_warranty_status(
  status: str,
  runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Record warranty status and transition to next step."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded: {status}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "warranty_status": status,
            "current_step": "specialist"  # Update state to trigger transition
        }
    )

# 3. Middleware applies dynamic configuration based on current_step
@wrap_model_call
def apply_step_config(
  request: ModelRequest,
  handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
  """Configure agent behavior based on current_step."""
  step = request.state.get("current_step", "triage")

  # Map steps to their configurations
  configs = {
    "triage": {
      "prompt": "Collect warranty information...",
      "tools": [record_warranty_status]
    },
    "specialist": {
      "prompt": "Provide solutions based on warranty: {warranty_status}",
      "tools": [provide_solution, escalate]
    }
  }

  config = configs[step]
  request = request.override(  
    system_prompt=config["prompt"].format(**request.state),  
    tools=config["tools"]  
  )
  return handler(request)

# 4. Create agent with middleware
agent = create_agent(
  model="gpt-4o",
  tools=[record_warranty_status, provide_solution, escalate],
  state_schema=SupportState,
  middleware=[apply_step_config],  
  checkpointer=InMemorySaver()  # Persist state across turns  #
)

# ------------------------------------
# Multiple agent subgraphs
# ------------------------------------

from typing import Literal
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import NotRequired

# 1. Define state with active_agent tracker
class MultiAgentState(AgentState):
  active_agent: NotRequired[str]

# 2. Create handoff tools
@tool
def transfer_to_sales(
  runtime: ToolRuntime,
) -> Command:
  """Transfer to the sales agent."""
  last_ai_message = next(  
    msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)  
  )  
  transfer_message = ToolMessage(  
    content="Transferred to sales agent from support agent",  
    tool_call_id=runtime.tool_call_id,  
  )  
  return Command(
    goto="sales_agent",
    update={
      "active_agent": "sales_agent",
      "messages": [last_ai_message, transfer_message],  
    },
    graph=Command.PARENT,
  )

@tool
def transfer_to_support(
  runtime: ToolRuntime,
) -> Command:
  """Transfer to the support agent."""
  last_ai_message = next(  
    msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)  
  )  
  transfer_message = ToolMessage(  
    content="Transferred to support agent from sales agent",  
    tool_call_id=runtime.tool_call_id,  
  )  
  return Command(
    goto="support_agent",
    update={
      "active_agent": "support_agent",
      "messages": [last_ai_message, transfer_message],  
    },
    graph=Command.PARENT,
  )

# 3. Create agents with handoff tools
sales_agent = create_agent(
  model="anthropic:claude-sonnet-4-20250514",
  tools=[transfer_to_support],
  system_prompt="You are a sales agent. Help with sales inquiries. If asked about technical issues or support, transfer to the support agent.",
)

support_agent = create_agent(
  model="anthropic:claude-sonnet-4-20250514",
  tools=[transfer_to_sales],
  system_prompt="You are a support agent. Help with technical issues. If asked about pricing or purchasing, transfer to the sales agent.",
)

# 4. Create agent nodes that invoke the agents
def call_sales_agent(state: MultiAgentState) -> Command:
  """Node that calls the sales agent."""
  response = sales_agent.invoke(state)
  return response

def call_support_agent(state: MultiAgentState) -> Command:
  """Node that calls the support agent."""
  response = support_agent.invoke(state)
  return response

# 5. Create router that checks if we should end or continue
def route_after_agent(
  state: MultiAgentState,
) -> Literal["sales_agent", "support_agent", "__end__"]:
  """Route based on active_agent, or END if the agent finished without handoff."""
  messages = state.get("messages", [])

  # Check the last message - if it's an AIMessage without tool calls, we're done
  if messages:
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:  
      return "__end__"

  # Otherwise route to the active agent
  active = state.get("active_agent", "sales_agent")
  return active if active else "sales_agent"

def route_initial(
  state: MultiAgentState,
) -> Literal["sales_agent", "support_agent"]:
  """Route to the active agent based on state, default to sales agent."""
  return state.get("active_agent") or "sales_agent"

# 6. Build the graph
builder = StateGraph(MultiAgentState)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)

# Start with conditional routing based on initial active_agent
builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])

# After each agent, check if we should end or route to another agent
builder.add_conditional_edges(
  "sales_agent", route_after_agent, ["sales_agent", "support_agent", END]
)
builder.add_conditional_edges(
  "support_agent", route_after_agent, ["sales_agent", "support_agent", END]
)

graph = builder.compile()
result = graph.invoke(
  {
    "messages": [
      {
        "role": "user",
        "content": "Hi, I'm having trouble with my account login. Can you help?",
      }
    ]
  }
)

for msg in result["messages"]:
  msg.pretty_print()

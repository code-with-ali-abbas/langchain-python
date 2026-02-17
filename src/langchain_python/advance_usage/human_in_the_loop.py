# ===========================================================================
# Human-in-the-Loop
# ===========================================================================

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
  model="gpt-4o",
  tools=[write_file_tool, execute_sql_tool, read_data_tool],
  middleware=[
    HumanInTheLoopMiddleware(
      interrupt_on={
        "write_file": True,                    # All decisions (approve, edit, reject) allowed
        "execute_sql": {"allowed_decisions": ["approve", "reject"]},    # No editing allowed
        # Safe operation, no approval needed
        "read_data": False,
      },
      description_prefix="Tool execution pending approval",
    )
  ],
  checkpointer=InMemorySaver()
)

# -----------------------------------------------
# Responding to interrupts
# -----------------------------------------------

from langgraph.types import Command

config = {"configurable": {"thread_id": "some_id"}} 

result = agent.invoke(
  {
    "messages": [
      {
        "role": "user",
        "content": "Delete old records from the database",
      }
    ]
  },
  config=config 
)

print(result['__interrupt__'])

agent.invoke(
  Command( 
    resume={"decisions": [{"type": "approve"}]}  # or "reject"
  ), 
  config=config # Same thread ID to resume the paused conversation
)

# -----------------------------------------------
# Decision types
# -----------------------------------------------

# ✅ approve

agent.invoke(
  Command(
    # Decisions are provided as a list, one per action under review.
    # The order of decisions must match the order of actions
    # listed in the `__interrupt__` request.
    resume={
      "decisions": [
        {
            "type": "approve",
        }
      ]
    }
  ),
  config=config  # Same thread ID to resume the paused conversation
)

# ✏️ edit

agent.invoke(
  Command(
    # Decisions are provided as a list, one per action under review.
    # The order of decisions must match the order of actions
    # listed in the `__interrupt__` request.
    resume={
      "decisions": [
        {
          "type": "edit",
          # Edited action with tool name and args
          "edited_action": {
            # Tool name to call.
            # Will usually be the same as the original action.
            "name": "new_tool_name",
            # Arguments to pass to the tool.
            "args": {"key1": "new_value", "key2": "original_value"},
          }
        }
      ]
    }
  ),
  config=config  # Same thread ID to resume the paused conversation
)

# ❌ reject

agent.invoke(
  Command(
    # Decisions are provided as a list, one per action under review.
    # The order of decisions must match the order of actions
    # listed in the `__interrupt__` request.
    resume={
      "decisions": [
        {
          "type": "reject",
          # An explanation about why the action was rejected
          "message": "No, this is wrong because ..., instead do this ...",
        }
      ]
    }
  ),
  config=config  # Same thread ID to resume the paused conversation
)

# -----------------------------------------------
# Multiple decisions
# -----------------------------------------------

{
  "decisions": [
    {"type": "approve"},
    {
      "type": "edit",
      "edited_action": {
        "name": "tool_name",
        "args": {"param": "new_value"}
      }
    },
    {
      "type": "reject",
      "message": "This action is not allowed"
    }
  ]
}

# -----------------------------------------------
# Streaming with human-in-the-loop
# -----------------------------------------------

config = {"configurable": {"thread_id": "some_id"}}

# Stream agent progress and LLM tokens until interrupt
for mode, chunk in agent.stream(
  {"messages": [{"role": "user", "content": "Delete old records from the database"}]},
  config=config,
  stream_mode=["updates", "messages"], 
):
  if mode == "messages":
    # LLM token
    token, metadata = chunk
    if token.content:
      print(token.content, end="", flush=True)
  elif mode == "updates":
    # Check for interrupt
    if "__interrupt__" in chunk:
      print(f"\n\nInterrupt: {chunk['__interrupt__']}")

# Resume with streaming after human decision
for mode, chunk in agent.stream(
  Command(resume={"decisions": [{"type": "approve"}]}),
  config=config,
  stream_mode=["updates", "messages"],
):
  if mode == "messages":
    token, metadata = chunk
    if token.content:
      print(token.content, end="", flush=True)

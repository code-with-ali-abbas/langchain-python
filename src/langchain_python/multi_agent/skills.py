# ===========================================================================
# Skills
# ===========================================================================

# ---------------------------------
# Basic implementation
# ---------------------------------

from langchain.tools import tool
from langchain.agents import create_agent

@tool
def load_skill(skill_name: str) -> str:
  """Load a specialized skill prompt.

  Available skills:
  - write_sql: SQL query writing expert
  - review_legal_doc: Legal document reviewer

  Returns the skill's prompt and context.
  """
  # Load skill content from file/database
  ...

agent = create_agent(
  model="gpt-4.1",
  tools=[load_skill],
  system_prompt=(
    "You are a helpful assistant. "
    "You have access to two skills: "
    "write_sql and review_legal_doc. "
    "Use load_skill to access them."
  ),
)

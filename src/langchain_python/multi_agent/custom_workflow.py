# ===========================================================================
# Custom Workflow
# ===========================================================================

# ---------------------------------
# Basic implementation
# ---------------------------------

from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END

agent = create_agent(model="openai:gpt-4.1", tools=[...])

def agent_node(state: State) -> dict:
  """A LangGraph node that invokes a LangChain agent."""
  result = agent.invoke({
    "messages": [{"role": "user", "content": state["query"]}]
  })
  return {"answer": result["messages"][-1].content}

# Build a simple workflow
workflow = (
  StateGraph(State)
  .add_node("agent", agent_node)
  .add_edge(START, "agent")
  .add_edge("agent", END)
  .compile()
)

# ---------------------------------
# Example: RAG pipeline
# ---------------------------------

from typing import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

class State(TypedDict):
  question: str
  rewritten_query: str
  documents: list[str]
  answer: str

# WNBA knowledge base with rosters, game results, and player stats
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_texts([
  # Rosters
  "New York Liberty 2024 roster: Breanna Stewart, Sabrina Ionescu, Jonquel Jones, Courtney Vandersloot.",
  "Las Vegas Aces 2024 roster: A'ja Wilson, Kelsey Plum, Jackie Young, Chelsea Gray.",
  "Indiana Fever 2024 roster: Caitlin Clark, Aliyah Boston, Kelsey Mitchell, NaLyssa Smith.",
  # Game results
  "2024 WNBA Finals: New York Liberty defeated Minnesota Lynx 3-2 to win the championship.",
  "June 15, 2024: Indiana Fever 85, Chicago Sky 79. Caitlin Clark had 23 points and 8 assists.",
  "August 20, 2024: Las Vegas Aces 92, Phoenix Mercury 84. A'ja Wilson scored 35 points.",
  # Player stats
  "A'ja Wilson 2024 season stats: 26.9 PPG, 11.9 RPG, 2.6 BPG. Won MVP award.",
  "Caitlin Clark 2024 rookie stats: 19.2 PPG, 8.4 APG, 5.7 RPG. Won Rookie of the Year.",
  "Breanna Stewart 2024 stats: 20.4 PPG, 8.5 RPG, 3.5 APG.",
])
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

@tool
def get_latest_news(query: str) -> str:
  """Get the latest WNBA news and updates."""
  # Your news API here
  return "Latest: The WNBA announced expanded playoff format for 2025..."

agent = create_agent(
  model="openai:gpt-4.1",
  tools=[get_latest_news],
)

model = ChatOpenAI(model="gpt-4.1")

class RewrittenQuery(BaseModel):
    query: str

def rewrite_query(state: State) -> dict:
  """Rewrite the user query for better retrieval."""
  system_prompt = """Rewrite this query to retrieve relevant WNBA information.
The knowledge base contains: team rosters, game results with scores, and player statistics (PPG, RPG, APG).
Focus on specific player names, team names, or stat categories mentioned."""
  response = model.with_structured_output(RewrittenQuery).invoke([
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": state["question"]}
  ])
  return {"rewritten_query": response.query}

def retrieve(state: State) -> dict:
  """Retrieve documents based on the rewritten query."""
  docs = retriever.invoke(state["rewritten_query"])
  return {"documents": [doc.page_content for doc in docs]}

def call_agent(state: State) -> dict:
  """Generate answer using retrieved context."""
  context = "\n\n".join(state["documents"])
  prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"
  response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
  return {"answer": response["messages"][-1].content_blocks}

workflow = (
  StateGraph(State)
  .add_node("rewrite", rewrite_query)
  .add_node("retrieve", retrieve)
  .add_node("agent", call_agent)
  .add_edge(START, "rewrite")
  .add_edge("rewrite", "retrieve")
  .add_edge("retrieve", "agent")
  .add_edge("agent", END)
  .compile()
)

result = workflow.invoke({"question": "Who won the 2024 WNBA Championship?"})
print(result["answer"])

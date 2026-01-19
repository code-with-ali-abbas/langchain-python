# ===========================================================================
# Structured Output
# ===========================================================================

# Provider Strategy

# Pydantic Model
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
  """Contact information for a person."""
  name: str = Field(description="The name of the person")
  email: str = Field(description="The email address of the person")
  phone: str = Field(description="The phone number of the person")

agent = create_agent(
  model="gpt-5",
  response_format=ContactInfo
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])

# Dataclass
from dataclasses import dataclass

@dataclass
class ContactInfo:
  """Contact information for a person."""
  name: str
  email: str
  phone: str

agent = create_agent(
  model="gpt-5",
  tools=[],
  response_format=ContactInfo
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]

# TypedDict
from typing_extensions import TypedDict

class ContactInfo(TypedDict):
  """Contact information for a person."""
  name: str
  email: str
  phone: str

agent = create_agent(
  model="gpt-5",
  tools=[],
  response_format=ContactInfo
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]

# JSON Schema
from langchain.agents.structured_output import ProviderStrategy

contact_info_schema = {
  "type": "object",
  "description": "Contact information for a person.",
  "properties": {
    "name": {"type": "string", "description": "The name of the person"},
    "email": {"type": "string", "description": "The email address of the person"},
    "phone": {"type": "string", "description": "The phone number of the person"}
  },
  "required": ["name", "email", "phone"]
}

agent = create_agent(
  model="gpt-5",
  tools=[],
  response_format=ProviderStrategy(contact_info_schema)
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]

# ===========================================================================
# Tool Calling Strategy
# ===========================================================================

# Pydantic Model
from typing import Literal
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
  """Analysis of a product review."""
  rating: int | None = Field(description="The rating of the product", ge=1, le=5)
  sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
  key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

agent = create_agent(
  model="gpt-5",
  tools=["tools"],
  response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]

# Dataclass

@dataclass
class ProductReview:
  """Analysis of a product review."""
  rating: int | None  # The rating of the product (1-5)
  sentiment: Literal["positive", "negative"]  # The sentiment of the review
  key_points: list[str]  # The key points of the review

agent = create_agent(
  model="gpt-5",
  tools=["tools"],
  response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]

# TypedDict

class ProductReview(TypedDict):
  """Analysis of a product review."""
  rating: int | None  # The rating of the product (1-5)
  sentiment: Literal["positive", "negative"]  # The sentiment of the review
  key_points: list[str]  # The key points of the review

agent = create_agent(
  model="gpt-5",
  tools=["tools"],
  response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]

# JSON Schema

product_review_schema = {
  "type": "object",
  "description": "Analysis of a product review.",
  "properties": {
    "rating": {
      "type": ["integer", "null"],
      "description": "The rating of the product (1-5)",
      "minimum": 1,
      "maximum": 5
    },
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative"],
      "description": "The sentiment of the review"
    },
    "key_points": {
      "type": "array",
      "items": {"type": "string"},
      "description": "The key points of the review"
    }
  },
  "required": ["sentiment", "key_points"]
}

agent = create_agent(
  model="gpt-5",
  tools=["tools"],
  response_format=ToolStrategy(product_review_schema)
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]

# Union Types

from typing import Union

class ProductReview(BaseModel):
  """Analysis of a product review."""
  rating: int | None = Field(description="The rating of the product", ge=1, le=5)
  sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
  key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

class CustomerComplaint(BaseModel):
  """A customer complaint about a product or service."""
  issue_type: Literal["product", "service", "shipping", "billing"] = Field(description="The type of issue")
  severity: Literal["low", "medium", "high"] = Field(description="The severity of the complaint")
  description: str = Field(description="Brief description of the complaint")

agent = create_agent(
  model="gpt-5",
  tools=["tools"],
  response_format=ToolStrategy(Union[ProductReview, CustomerComplaint])
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]

# ===========================================================================
# Custom Tool Message
# ===========================================================================

class MeetingAction(BaseModel):
  """Action items extracted from a meeting transcript."""
  task: str = Field(description="The specific task to be completed")
  assignee: str = Field(description="Person responsible for the task")
  priority: Literal["low", "medium", "high"] = Field(description="Priority level")

agent = create_agent(
  model="gpt-5",
  tools=[],
  response_format=ToolStrategy(
    schema=MeetingAction,
    tool_message_content="Action item captured and added to meeting notes!"
  )
)

agent.invoke({
  "messages": [{"role": "user", "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"}]
})

# ===========================================================================
# Schema Validation Error
# ===========================================================================

class ProductRating(BaseModel):
  rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
  comment: str = Field(description="Review comment")

agent = create_agent(
  model="gpt-5",
  tools=[],
  response_format=ToolStrategy(ProductRating),  # Default: handle_errors=True
  system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up."
)

agent.invoke({
  "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]
})

# ===========================================================================
# Error Handling Strategies
# ===========================================================================

# Custom error message:
ToolStrategy(
  schema=ProductRating,
  handle_errors="Please provide a valid rating between 1-5 and include a comment."
)

# Handle specific exceptions only
ToolStrategy(
  schema=ProductRating,
  handle_errors=ValueError
)

# Handle multiple exception types
ToolStrategy(
  schema=ProductRating,
  handle_errors=(ValueError, TypeError)  # Retry on ValueError and TypeError
)

# Custom error handler function
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError

def custom_error_handler(error: Exception) -> str:
  if isinstance(error, StructuredOutputValidationError):
    return "There was an issue with the format. Try again."
  elif isinstance(error, MultipleStructuredOutputsError):
    return "Multiple structured outputs were returned. Pick the most relevant one."
  else:
    return f"Error: {str(error)}"
  
agent = create_agent(
  model="gpt-5",
  tools=[],
  response_format=ToolStrategy(
    schema=Union[ProductReview, CustomerComplaint],
    handle_errors=custom_error_handler
  )
)

result = agent.invoke({
  "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})

for msg in result['messages']:
  # If message is actually a ToolMessage object (not a dict), check its class name
  if type(msg).__name__ == "ToolMessage":
    print(msg.content)
  # If message is a dictionary or you want a fallback
  elif isinstance(msg, dict) and msg.get('tool_call_id'):
    print(msg['content'])

# No error handling
response_format = ToolStrategy(
  schema=ProductRating,
  handle_errors=False
)

from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field

from main.model_factory import model_factory


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b
class add(BaseModel):
    """Add two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")
tools = [add, multiply]
llm = model_factory().create_model()
llm_with_tools = llm.bind_tools(tools)


query = "What is 3 * 12?"

chain = llm_with_tools | PydanticToolsParser(tools=[add, multiply])
chain.invoke(query)

llm_with_tools.invoke(query)
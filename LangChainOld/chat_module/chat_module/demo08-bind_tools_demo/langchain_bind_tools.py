from pydantic import Field, BaseModel

from model_factory.model_factory import model_factory


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

#  工具调用，ToolCall 是一个包含 工具名称、参数值字典和（可选）标识符的类型字典
var = llm_with_tools.invoke(query).tool_calls
print(var)

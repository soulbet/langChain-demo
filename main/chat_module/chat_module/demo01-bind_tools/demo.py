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

tools = [add, multiply]
llm = model_factory().create_model()
# ool_choice="Multiply" 强制使用工具
# parallel_tool_calls=False  仅调用一个工具一次
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")


query = "What is 3 * 12?"
result = llm_with_tools.invoke(query)
print(result.tool_calls)


from typing import List

from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict


class Args(TypedDict):
    a: int
    b: List[int]


def f(x: Args) -> str:
    return str(x["a"] * max(x["b"]))


runnable = RunnableLambda(f)
as_tool = runnable.as_tool(
    name="My tool",
    description="Explanation of when to use tool.",
)
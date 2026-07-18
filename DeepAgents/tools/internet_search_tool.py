import os
from typing import Literal

from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from deepagents import create_deep_agent

from model_factory import model_factory

tavily_client = TavilyClient("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")


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
llm = model_factory().create_model("qwen")

agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    system_prompt="请使用工具 `internet_search`查询"
)
result = agent.invoke({"messages":[HumanMessage(content="""
你首先需要查一下今天的日期，然后查上海今天天气怎么样
""")]})

last_message = result["messages"][-1]
# 如果是 AIMessage 对象，使用 .content
if hasattr(last_message, "content"):
    content = last_message.content
    print(content)
    # 或者进一步处理字符串
total_input = 0
total_output = 0

for msg in result.get("messages", []):
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        total_input += msg.usage_metadata.get("input_tokens", 0)
        total_output += msg.usage_metadata.get("output_tokens", 0)

print(f"总输入 tokens: {total_input}")
print(f"总输出 tokens: {total_output}")

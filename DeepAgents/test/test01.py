from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

from model_factory import model_factory
tavily_client = TavilyClient("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")

llm = model_factory().create_model("qwen")

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
# 使用本地 Shell 后端
backend = LocalShellBackend(root_dir=r"D:\workspace\python\python_demo\langChain-demo")

agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    backend=backend,  # 注入后端
    system_prompt="你是一个文件操作助手。"
)

# Agent 现在可以使用 ls, read_file, write_file 等工具
result = agent.invoke({
    "messages": [HumanMessage(content="列出当前目录下的所有文件")]
})
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
from typing import Literal
# from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage

from model_factory.custom_factory import DebugChatModel
from model_factory.model_factory import model_factory




def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""

    tavily_client = model_factory().access_data_from_tavily()
    result = tavily_client.search(
        query=query,
        max_results=3,
        search_depth="basic",
        topic="general"
    )
    # print(result)
    # print(len(result))
    # # 如果结果太多，可以只提取标题和简短摘要
    # if isinstance(result, dict) and "results" in result:
    #     summary = []
    #     for r in result["results"][:max_results]:
    #         summary.append(f"- {r.get('title')}: {r.get('content', '')[:200]}...")
    #
    #     return "\n".join(summary) if summary else "未找到相关结果"

    return result


llm=model_factory().create_model()
debug_llm=DebugChatModel(real_model=llm)
agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    system_prompt="请结合工具`internet_search`回答问题"
)
for chunk in agent.stream(
    {"messages": [HumanMessage(content="AI的最新动向")]},
    stream_mode="messages",
    version="v2"
):
    # 检查消息类型，找到系统提示词
    if chunk["type"] == "messages":
        # 这里的 data 通常是 (message, metadata) 的元组
        # 如果消息角色是 'system'，那它就是我们要找的组装后的提示词
        message, metadata = chunk["data"]
        if hasattr(message, 'role') and message.role == "system":
            print("=== 完整的系统提示词 ===")
            print(message.content)
            print("=== 结束 ===")
            break
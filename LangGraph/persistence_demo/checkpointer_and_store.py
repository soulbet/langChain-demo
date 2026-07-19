from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.channels.delta import DeltaChannel
from typing import Annotated, List

from model_factory.model_factory import model_factory


# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city} 天气晴朗，25°C"

# 2. 定义状态（使用 DeltaChannel 优化）
class AgentState(MessagesState):
    # 消息使用 DeltaChannel，每 30 步保存一次完整快照
    messages: Annotated[List, DeltaChannel(
        snapshot_frequency=30
    )]
    # 额外状态
    current_city: str

# 3. 构建图
builder = StateGraph(AgentState)
llm = model_factory().create_model()

# 定义节点
def call_model(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([get_weather]))
builder.add_edge("agent", "tools", condition=tools_condition)
builder.set_entry_point("agent")

# 4. 编译时传入检查点（配置保存时机）
checkpointer = MemorySaver()  # 测试用，生产换 PostgresSaver
graph = builder.compile(checkpointer=checkpointer)

# 5. 使用：指定 thread_id
config = {"configurable": {"thread_id": "user-123"}}

# 第一次对话
result1 = graph.invoke(
    {"messages": [("user", "上海天气怎么样？")]},
    config=config
)
print(result1["messages"][-1].content)

# 第二次对话（Agent 记得之前内容）
result2 = graph.invoke(
    {"messages": [("user", "那里需要带伞吗？")]},
    config=config  # 同一个 thread_id，自动从检查点恢复
)
print(result2["messages"][-1].content)
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, filter_messages
)

from main.model_factory import model_factory

"""
filter_messages参数：
include_types / exclude_types	字符串或BaseMessage类	按消息类型过滤，如 "human", "ai", "system", "tool"。
include_names / exclude_names	字符串序列	按消息的 name 属性过滤，适合在多用户或不同角色间筛选。
include_ids / exclude_ids	字符串序列	按消息的唯一 id 进行精确过滤，适合移除或保留特定消息。
exclude_tool_calls	布尔值或字符串序列	高级功能。排除包含特定工具调用ID的消息或工具调用本身。
"""

messages = [
    SystemMessage("你是一个乐于助人的助手。", id="1"),
    HumanMessage("我的名字是什么？", id="2", name="user1"),
    AIMessage("抱歉，我没有您的个人信息。", id="3", name="assistant"),
    HumanMessage("我叫小明。", id="4", name="user1"),
    AIMessage("好的，小明，很高兴认识你！", id="5", name="assistant"),
]

# 只保留 HumanMessage 和 AIMessage
filtered = filter_messages(
    messages,
    include_types=[HumanMessage, AIMessage]  # 也支持字符串 "human", "ai"
)
# 排除名为 "example_user" 和 "example_assistant" 的消息
filtered1 = filter_messages(
    messages,
    exclude_names=["example_user", "example_assistant"]
)
# 排除 ID 为 "3" 的消息
filtered2 = filter_messages(
    messages,
    include_types=[HumanMessage, AIMessage],  # 先包含这两类
    exclude_ids=["3"]  # 再从中排除 ID 为 3 的
)
llm = model_factory().create_model()
print(filtered)
## 消息会先经过过滤器，只有保留下来的消息才会被发送给模型，从而实现了对输入的精准控制
chain = filtered | llm
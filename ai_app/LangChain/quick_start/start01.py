from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage


@tool
def save_user_preference(preference: str, runtime: ToolRuntime) -> str:
    """保存用户的偏好设置。"""
    # 1. 从上下文中获取用户 ID
    msg = runtime.state["messages"]
    user_id = runtime.context.get("user_id", "anonymous")

    # 2. 保存到长期存储
    runtime.store.set(f"pref_{user_id}", preference)

    # 3. 从状态中读取历史消息
    last_user_msg = None
    for msg in reversed(runtime.state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    return f"已为用户 {user_id} 保存偏好：'{preference}'。上一条用户消息是：'{last_user_msg}'"


# 在调用 Agent 时，通过 context 参数传入数据：
agent.invoke(
    {"messages": [{"role": "user", "content": "我喜欢短的回答"}]},
    config={"configurable": {"thread_id": "123"}},
    context={"user_id": "user_456"}  # 👈 传入上下文
)
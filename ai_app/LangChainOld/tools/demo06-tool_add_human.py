from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langchain.tools import tool

# 1. 定义高风险工具
@tool
def delete_file(path: str) -> str:
    """Delete a file from the filesystem."""
    return f"Deleted {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to}"

# 2. 必须配置检查点
checkpointer = MemorySaver()

# 3. 创建 Agent 并配置中断
agent = create_deep_agent(
    model="your_model",
    tools=[delete_file, send_email],
    interrupt_on={
        "delete_file": True,  # 默认允许：approve, edit, reject[citation:1]
        "send_email": {"allowed_decisions": ["approve", "reject"]}  # 只允许同意或拒绝
    },
    checkpointer=checkpointer  # 必须！
)

# 4. 设置线程 ID（用于恢复会话）
config = {"configurable": {"thread_id": "user_session_123"}}

# 5. 用户发起请求
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete the file temp.txt and send an email to admin."}]},
    config=config,
)

# 6. 检查是否被中断
if result.get('__interrupt__'):
    interrupt_value = result['__interrupt__'][0].value
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]

    # 7. 向用户展示待审批的动作
    print("等待审批的动作:")
    for action in action_requests:
        print(f"  - 工具: {action['name']}, 参数: {action['args']}")

    # 8. 模拟人工决策（在实际应用中，这里会是用户界面或接口）
    # 按顺序提供决策，与 action_requests 顺序对应[citation:1][citation:2]
    decisions = [
        {"type": "approve"},  # 同意删除文件
        {"type": "reject"}    # 拒绝发送邮件
    ]

    # 9. 使用 Command 恢复执行
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,  # 必须使用相同的 config!
    )

# 10. 处理最终结果
print(result["messages"][-1].content)
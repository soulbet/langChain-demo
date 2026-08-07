from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend


# 1. 创建一个沙箱客户端
sandbox_backend = LocalShellBackend()

# 2. 用于存储映射关系的字典（生产环境应使用 Redis 或数据库）
sandbox_store = {}

def get_or_create_sandbox(thread_id: str):
    """根据 thread_id 获取或创建沙箱"""
    if thread_id not in sandbox_store:
        # 首次运行：创建沙箱
        sandbox = sandbox_backend.create_sandbox(
            metadata={"thread_id": thread_id, "created_at": "2024-01-01"}
        )
        sandbox_store[thread_id] = sandbox.sandbox_id
        return sandbox
    else:
        # 后续轮次：通过 ID 重新连接已有的沙箱
        sandbox_id = sandbox_store[thread_id]
        return client.connect_sandbox(sandbox_id)

# 3. 在 Agent 中使用沙箱
thread_id = "user-123"
sandbox = get_or_create_sandbox(thread_id)

backend = SandboxBackend(sandbox=sandbox)

agent = create_deep_agent(
    model=llm,
    backend=backend,
)

# 第一次调用：创建沙箱
result1 = agent.invoke(
    {"messages": [HumanMessage(content="创建一个名为 test.txt 的文件")]},
    config={"configurable": {"thread_id": thread_id}}
)

# 第二次调用（同一线程）：复用沙箱，文件仍然存在
result2 = agent.invoke(
    {"messages": [HumanMessage(content="读取 test.txt 的内容")]},
    config={"configurable": {"thread_id": thread_id}}
)
# 此时，沙箱被重用，test.txt 依然存在
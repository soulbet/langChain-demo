from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.messages import HumanMessage

from model_factory.model_factory import model_factory

# 配置沙箱后端
sandbox_backend = LocalShellBackend(
    root_dir="./",  # 沙箱内的工作目录
    # 可选：配置资源限制、网络访问等
)
llm = model_factory().create_model("qwen")
agent = create_deep_agent(
    model=llm,
    backend=sandbox_backend,  # 使用沙箱后端
    system_prompt="你是一个可以在沙箱中执行代码和命令的助手。"
)

# Agent 现在可以使用 execute 工具运行 Shell 命令
result = agent.invoke({
    "messages": [HumanMessage(content="在沙箱中运行 `ls -la` 列出文件")]
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
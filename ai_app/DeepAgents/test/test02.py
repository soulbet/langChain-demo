from deepagents.backends import FilesystemBackend
import deepagents
from langchain_core.messages import HumanMessage, SystemMessage

from ai_app.model_factory.model_factory import ModelFactory


llm = ModelFactory().create_model(
    local_model_type="agent"
)

system_prompt = SystemMessage(
    content="""
你是一个文件系统助手。

当用户要求查看文件或目录时，
必须使用文件系统工具。

不要猜测文件内容。
"""
)

backend = FilesystemBackend(
    root_dir=r"D:\work_space\python\langChain-demo",
    virtual_mode=True,
)

agent = deepagents.create_deep_agent(
    llm,
    system_prompt=system_prompt,
    backend=backend,
)

response = agent.invoke({
    "messages": [
        HumanMessage(
            content="请使用文件系统工具列出当前目录。"
        )
    ]
})

for i, message in enumerate(response["messages"]):
    print(f"\n========== MESSAGE {i} ==========")
    print(type(message).__name__)
    print("CONTENT:")
    print(message.content)

    if hasattr(message, "tool_calls"):
        print("TOOL_CALLS:")
        print(message.tool_calls)
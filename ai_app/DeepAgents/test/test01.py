from deepagents.backends import FilesystemBackend
from langchain_community.callbacks import get_openai_callback
import deepagents
from langchain_core.messages import SystemMessage, HumanMessage

from ai_app.model_factory.model_factory import ModelFactory
from langchain_core.callbacks import BaseCallbackHandler

# class DebugHandler(BaseCallbackHandler):
#
#     def __init__(self):
#         self.call_count = 0
#
#     def on_chat_model_start(
#         self,
#         serialized,
#         messages,
#         **kwargs
#     ):
#         self.call_count += 1
#
#         print("\n")
#         print("#" * 100)
#         print(f"MODEL CALL #{self.call_count}")
#         print("#" * 100)
#
#         print("\nRAW MESSAGES:")
#         print(repr(messages))
#
#         print("\nKWARGS:")
#         print(repr(kwargs))
#
#         print("\nSERIALIZED:")
#         print(repr(serialized))


# =========================
# 1. 模型
# =========================

llm = ModelFactory().create_model(
    local_model_type="agent"
)

# =========================
# 2. System Prompt
# =========================

system_prompt = """
你是一个代码项目分析助手。

当前文件系统的根目录就是项目根目录。

当用户要求查看文件或目录时，必须调用文件系统工具。
不要假装调用工具。
不要猜测文件内容。
"""

# =========================
# 3. Windows 文件系统
# =========================

backend = FilesystemBackend(
    root_dir=r"D:\work_space\python\langChain-demo",
    virtual_mode=True,
)

# =========================
# 4. 创建 Agent
# =========================

agent = deepagents.create_deep_agent(
    llm,
    system_prompt=system_prompt,
    backend=backend,
)

# =========================
# 5. 最小测试
# =========================

human_prompt ={
    "messages": [
        HumanMessage(
            content="请使用文件系统工具列出当前目录，然后告诉我目录中有哪些文件和文件夹。"
        )
    ]
}
with get_openai_callback() as cb:
    response = agent.invoke(human_prompt)

    for i, message in enumerate(response["messages"]):
        print("CONTENT:")
        print(message.content)


    print("\n📊 Token 使用统计：")
    print(f"  总 tokens: {cb.total_tokens}")
    print(f"  输入 tokens: {cb.prompt_tokens}")
    print(f"  输出 tokens: {cb.completion_tokens}")

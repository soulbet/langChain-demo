from abc import ABC
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


class DebugChatModel(BaseChatModel, ABC):
    """一个只打印消息而不实际调用模型的包装器"""
    real_model: BaseChatModel  # 你真正的模型实例

    @property
    def _llm_type(self) -> str:
        """返回模型类型标识，用于调试"""
        return "debug_chat_model"
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs,
    ) -> ChatResult:
        # 1. 打印完整的消息列表
        print("\n" + "=" * 80)
        print("📨 准备发送给模型的完整消息:")
        print("=" * 80)
        for msg in messages:
            print(f"\n[角色: {msg.type}]")
            print(f"内容:\n{msg.content}")
            print("-" * 40)
        print("=" * 80)

        # 2. 这里是关键：如果你希望看到提示词后停止，就注释掉下面这行
        # 如果你希望继续实际调用模型，则取消注释，但会继续报错
        # return self.real_model._generate(messages, stop=stop, **kwargs)

        # 3. 如果只是想看提示词，返回一个空结果
        return ChatResult(generations=[])
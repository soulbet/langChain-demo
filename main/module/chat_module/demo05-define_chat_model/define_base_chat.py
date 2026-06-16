"""
BaseChatModel可以实现一下内容；

_generate	用于从提示生成聊天结果	必需
_llm_type (属性)	用于唯一标识模型的类型。用于日志记录。	必需
_identifying_params (属性)	表示用于追踪目的的模型参数化。	可选
_stream	用于实现流式处理。	可选
_agenerate	用于实现原生异步方法。	可选
_astream	用于实现 _stream 的异步版本。	可选

"""


from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor


class CustomChatModelAdvanced(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str # 模型名称（用于标识）

    n: int # 从输入中提取的字符数

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """实现模型的核心生成逻辑（非流式调用）。
        """
        # 1. 提取最后一条用户消息
        last_message = messages[-1]
        # 2. 提取前 n 个字符作为响应
        tokens = last_message.content[: self.n]
        # 3. 创建 AIMessage（AI 的响应消息）
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
        )

        # 4. 包装成 ChatGeneration 和 ChatResult 返回
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """- 流式生成
        实现流式输出，逐个 token 返回结果。
        """
        # 1. 提取响应字符串
        last_message = messages[-1]
        tokens = last_message.content[: self.n]

        # 2. 逐个字符 yield 出去
        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # 3. 最后额外添加元数据
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        # 模型类型标识
        """用于回调系统和日志记录，标识这是哪种类型的 LLM。"""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """- 识别参数
        用于监控、追踪（如 LangSmith）和缓存，帮助系统识别不同的模型实例。
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }
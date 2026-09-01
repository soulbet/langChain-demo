# -*- coding: utf-8 -*-
"""
middleware_demo.py
===================

LangChain Agent 中间件（Middleware）全部 hooks 的演示与参考实现。

运行前请确保版本：``langchain>=1.0``、``langchain-core>=1.4``
（本项目环境为 langchain 1.3.14 / langchain-core 1.4.9）。

本文件包含两部分：
    1. :class:`DemoMiddleware` —— 继承 ``AgentMiddleware``，把每一个 hook 都实现一遍。
    2. ``main()`` —— 用一个伪模型 + 一个工具 + 中间件跑一次 agent，
       观察 hook 的实际调用顺序（可用于验收、学习）。

直接运行::

    python ai_app/LangChain/middleware/middleware_demo.py
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class BindableFakeChatModel(FakeMessagesListChatModel):
    """伪模型：补上 bind_tools 支持，让 agent 能正常构建。"""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


# ----------------------------------------------------------------------
# 一个被 agent 调用的普通工具（用于触发 wrap_tool_call / tools 节点）
# ----------------------------------------------------------------------
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# ======================================================================
# 第一种写法：子类化 AgentMiddleware，把每个 hook 都实现一遍
# ======================================================================
class DemoMiddleware(AgentMiddleware):
    """实现了全部 hooks 的示例中间件。

    hook 一览（每个都有 同步 / 异步 两个版本）:
        before_agent   : Agent 开始执行前
        before_model   : 每次调用模型前
        wrap_model_call: 包裹/拦截模型调用
        after_model    : 每次调用模型后
        wrap_tool_call : 包裹/拦截工具调用
        after_agent    : Agent 执行结束后（整个 run 只跑一次）

    类属性:
        state_schema : 自定义的 state schema（默认 AgentState）
        tools        : 该中间件额外注册的工具
        transformers : 流式 transformer 工厂
    """

    # 可用自定义 state schema；这里用默认即可
    # state_schema = AgentState

    # 可选：中间件额外提供的工具（会合并进 agent 的工具集）
    tools = ()

    @property
    def name(self) -> str:
        return "DemoMiddleware"

    # ----- 1. before_agent -----
    def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        print("[before_agent] agent 即将开始, 消息数 =", len(state["messages"]))
        return None  # 可返回 dict 注入 state，或返回 {"jump_to": "end", ...} 提前结束

    async def abefore_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        print("[abefore_agent] (async)")
        return None

    # ----- 2. before_model -----
    @hook_config(can_jump_to=["tools", "model", "end"])
    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        print("[before_model] 即将调用模型")
        return None

    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        print("[abefore_model] (async)")
        return None

    # ----- 3. wrap_model_call（包裹模型调用）-----
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> Any:
        print("[wrap_model_call] 拦截模型调用 ->", request.model.__class__.__name__)
        # 可多次调用 handler 实现重试；也可不调用它来短路返回
        response = handler(request)
        print("[wrap_model_call] 模型返回 message 数 =", len(response.result))
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        print("[awrap_model_call] (async)")
        return await handler(request)

    # ----- 4. after_model -----
    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1]
        print("[after_model] 模型已响应 ->", last.__class__.__name__)
        return None

    async def aafter_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        print("[aafter_model] (async)")
        return None

    # ----- 5. wrap_tool_call（包裹工具调用）-----
    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage],
    ) -> ToolMessage:
        name = request.tool_call.get("name")
        print("[wrap_tool_call] 拦截工具调用 ->", name)
        result = handler(request)
        print("[wrap_tool_call] 工具返回 ->", result.content)
        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        print("[awrap_tool_call] (async)")
        return await handler(request)

    # ----- 6. after_agent -----
    def after_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        print("[after_agent] agent 执行结束, 消息数 =", len(state["messages"]))
        return None

    async def aafter_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        print("[aafter_agent] (async)")
        return None


# ======================================================================
# 第二种写法：用装饰器把「单个函数」变成中间件（无需子类化）
# ======================================================================
@dynamic_prompt
def my_dynamic_prompt(request: ModelRequest[Any]) -> str:
    """动态生成 system prompt。"""
    n = len(request.state["messages"])
    return f"你是测试助手。当前已有 {n} 条消息。"


@before_agent
def log_before_agent(state: AgentState, runtime: Any) -> None:
    print("[装饰器 before_agent] start")


@after_agent
def log_after_agent(state: AgentState, runtime: Any) -> None:
    print("[装饰器 after_agent] done")


@before_model
def log_before_model(state: AgentState, runtime: Any) -> None:
    print("[装饰器 before_model] about to model")


# 注意：@wrap_model_call / @wrap_tool_call / @after_model 用法与此类似，
# 直接参考上方类的同名方法即可。装饰器方式适合「只关心单个 hook」的轻量场景。


# ======================================================================
# 运行演示
# ======================================================================
def main() -> None:
    # ---- 伪模型：第一轮返回工具调用，第二轮给出最终答案 ----
    model = BindableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "add",
                        "args": {"a": 3, "b": 5},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="结果是 8。"),
        ]
    )

    # use_messages: 伪模型不在乎 messages，只按 responses 依次吐出
    # ---- 用类式中间件 + 装饰器式中间件组装 agent ----
    agent = create_agent(
        model,
        tools=[add],
        middleware=[
            DemoMiddleware(),       # 类式：实现全部 hooks
            my_dynamic_prompt,      # 装饰器式：dynamic_prompt
            log_before_agent,
            log_after_agent,
            log_before_model,
        ],
    )

    print("=" * 60)
    print("开始执行 agent，观察 hook 调用顺序：")
    print("=" * 60)
    result = agent.invoke({"messages": [{"role": "user", "content": "3 + 5 = ?"}]})
    print("=" * 60)
    print("最终消息:", result["messages"][-1].content)
    print("=" * 60)


if __name__ == "__main__":
    main()

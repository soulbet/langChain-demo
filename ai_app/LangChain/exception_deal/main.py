#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/9/2 16:55
@Author : zyf
@File : main.py
@Project : langChain-demo
@Software : PyCharm
@explain : LangChain/LangGraph 异常处理完整示例
@DESCRIPTION :
    覆盖 Agent 全链路的 6 类异常场景及处理方案：
    1. 模型调用异常 —— 超时、限流(429)、网络错误、服务不可用
    2. 工具调用异常 —— 工具执行报错、参数错误、handle_tool_error 机制
    3. 重试机制      —— RetryPolicy 指数退避、自定义重试条件
    4. Fallback 降级 —— 主模型不可用时自动切换备用模型
    5. LCEL 链异常  —— 管道中的异常传播与捕获
    6. 图级别异常    —— LangGraph 节点异常处理与全局错误恢复

    依赖：langgraph>=1.2.7, langchain>=1.3.11, langchain-core>=1.4.8
"""

import time
import random
from typing import Annotated, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from ai_app.model_factory.model_factory import model_factory


# ======================================================================
# 1. 模拟各类异常的工具
# ======================================================================

@tool
def unreliable_api_call(query: str) -> str:
    """调用一个不稳定的外部 API（模拟随机失败）。"""
    if random.random() < 0.5:
        raise ConnectionError(f"API 调用失败：连接超时，query={query}")
    return f"API 返回结果：'{query}' 的查询成功，找到 3 条记录。"


@tool
def fragile_tool(x: int) -> str:
    """一个容易出错的工具（模拟各种异常）。"""
    if x < 0:
        raise ValueError(f"参数错误：x 不能为负数，当前值: {x}")
    if x == 0:
        raise ZeroDivisionError("除零错误")
    return f"计算结果: {100 / x}"


@tool
def handle_tool_error_demo(x: int) -> str:
    """演示 handle_tool_error 的工具，参数为负数时会抛异常。"""
    if x < 0:
        raise ValueError(f"不支持负数输入: {x}")
    return f"处理成功: {x * 2}"


@tool
def timeout_tool(seconds: int) -> str:
    """模拟超时操作。"""
    if seconds > 5:
        time.sleep(seconds)
    return f"操作完成，耗时 {seconds} 秒。"


@tool
def get_weather(city: str) -> str:
    """获取城市天气信息（正常工具）。"""
    return f"{city} 天气晴朗，气温 25°C，湿度 60%。"


# ======================================================================
# 2. 定义状态
# ======================================================================
class AgentState(MessagesState):
    messages: Annotated[List, list]
    error_count: int = 0
    last_error: str = ""


# ======================================================================
# 3. 演示一：工具调用异常 —— handle_tool_error
# ======================================================================
def demo_tool_error_handling():
    """
    handle_tool_error 机制：
    工具抛出异常时，不会导致整个 Agent 崩溃，
    而是将错误信息作为 ToolMessage 返回给 LLM，让 LLM 自行修正。
    """
    print("\n" + "=" * 70)
    print("演示一：工具调用异常 —— handle_tool_error 机制")
    print("=" * 70)

    model = model_factory().create_model()

    # 方式一：handle_tool_error=True（返回通用错误消息）
    fragile_tool_with_error = fragile_tool.copy()
    fragile_tool_with_error.handle_tool_error = True

    # 方式二：handle_tool_error=自定义消息
    handle_tool_error_demo_with_error = handle_tool_error_demo.copy()
    handle_tool_error_demo_with_error.handle_tool_error = "工具执行出错，请检查参数后重试。"

    # 方式三：handle_tool_error=回调函数（根据异常类型返回不同消息）
    def custom_error_handler(e):
        if isinstance(e, ValueError):
            return f"参数校验失败: {e}。请确保传入正整数。"
        elif isinstance(e, ZeroDivisionError):
            return "不能除以零，请换一个非零的数字。"
        else:
            return f"未知错误: {type(e).__name__}: {e}"

    fragile_tool_custom = fragile_tool.copy()
    fragile_tool_custom.handle_tool_error = custom_error_handler

    tools = [fragile_tool_custom, get_weather]
    llm_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_error=custom_error_handler)

    def call_model(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "demo-tool-error-001"}}

    print("\n--- 测试：传入负数触发 ValueError ---")
    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我计算 fragile_tool 传入 -5 的结果")]},
        config=config,
    )
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  [{msg.name}] {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"  [AI] {msg.content[:200]}")


# ======================================================================
# 4. 演示二：RetryPolicy —— 节点级重试（指数退避）
# ======================================================================
def demo_retry_policy():
    """
    RetryPolicy：LangGraph 节点级重试机制。
    支持指数退避、自定义重试条件、最大重试次数。
    """
    print("\n" + "=" * 70)
    print("演示二：RetryPolicy —— 节点级重试（指数退避）")
    print("=" * 70)

    model = model_factory().create_model()
    call_count = {"count": 0}

    def flaky_api_call(state: AgentState) -> dict:
        """模拟不稳定的 API 调用节点（前两次失败，第三次成功）。"""
        call_count["count"] += 1
        print(f"  [API 调用] 第 {call_count['count']} 次尝试...")

        if call_count["count"] < 3:
            raise ConnectionError(f"连接失败（第 {call_count['count']} 次）")

        return {"messages": [AIMessage(content="API 调用成功！获取到数据。")]}

    # RetryPolicy 配置：
    # - max_attempts: 最大尝试次数
    # - retry_on: 哪些异常触发重试（可以是异常类型元组或函数）
    # - initial_interval: 初始重试间隔（秒）
    # - max_interval: 最大重试间隔（秒）
    retry_policy = RetryPolicy(
        max_attempts=5,
        retry_on=(ConnectionError, TimeoutError),
        initial_interval=0.1,
        max_interval=2.0,
    )

    builder = StateGraph(AgentState)
    builder.add_node("api_call", flaky_api_call, retry=retry_policy)
    builder.add_edge(START, "api_call")
    builder.add_edge("api_call", END)

    graph = builder.compile()
    result = graph.invoke({"messages": []})

    print(f"\n--- 结果 ---")
    print(f"  总尝试次数: {call_count['count']}")
    print(f"  最终结果: {result['messages'][-1].content}")


# ======================================================================
# 5. 演示三：自定义重试条件函数
# ======================================================================
def demo_custom_retry_condition():
    """
    retry_on 可以传函数，根据异常内容动态决定是否重试。
    典型场景：只对 429(限流) 和 502/503/504(网关错误) 重试。
    """
    print("\n" + "=" * 70)
    print("演示三：自定义重试条件函数")
    print("=" * 70)

    attempt_count = {"count": 0}

    def simulate_rate_limited_call(state: AgentState) -> dict:
        """模拟被限流的 API 调用。"""
        attempt_count["count"] += 1
        print(f"  [调用] 第 {attempt_count['count']} 次...")

        if attempt_count["count"] <= 2:
            error = Exception("HTTP 429 Too Many Requests")
            error.response = type("Response", (), {"status_code": 429})()
            raise error
        return {"messages": [AIMessage(content="限流解除，调用成功！")]}

    def should_retry(error: Exception) -> bool:
        """只对特定 HTTP 状态码重试。"""
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            code = error.response.status_code
            print(f"  [重试判断] HTTP {code} → {'重试' if code in [429, 502, 503, 504] else '不重试'}")
            return code in [429, 502, 503, 504]
        return False

    retry_policy = RetryPolicy(
        max_attempts=5,
        retry_on=should_retry,
        initial_interval=0.1,
    )

    builder = StateGraph(AgentState)
    builder.add_node("api", simulate_rate_limited_call, retry=retry_policy)
    builder.add_edge(START, "api")
    builder.add_edge("api", END)

    graph = builder.compile()
    result = graph.invoke({"messages": []})

    print(f"\n--- 结果 ---")
    print(f"  总尝试次数: {attempt_count['count']}")
    print(f"  最终结果: {result['messages'][-1].content}")


# ======================================================================
# 6. 演示四：Fallback 降级 —— 主模型失败自动切换备用
# ======================================================================
def demo_fallback_model():
    """
    with_fallbacks：主模型调用失败时，自动切换到备用模型。
    典型场景：主模型限流/宕机 → 切换到备用模型继续服务。
    """
    print("\n" + "=" * 70)
    print("演示四：Fallback 降级 —— 主模型失败自动切换备用")
    print("=" * 70)

    primary_model = model_factory().create_model()
    fallback_model = model_factory().create_model()

    # 模拟主模型不稳定（包装一层会随机失败的调用）
    call_log = []

    def primary_call(state: AgentState) -> dict:
        call_log.append("primary")
        print(f"  [调用] 尝试主模型...")
        if len(call_log) <= 1:
            raise ConnectionError("主模型服务不可用")
        response = primary_model.invoke(state["messages"])
        return {"messages": [response]}

    def fallback_call(state: AgentState) -> dict:
        call_log.append("fallback")
        print(f"  [调用] 切换到备用模型...")
        response = fallback_model.invoke(state["messages"])
        return {"messages": [response]}

    # 方式一：在 LangGraph 节点中用 try-except 实现 fallback
    def agent_with_fallback(state: AgentState) -> dict:
        try:
            return primary_call(state)
        except Exception as e:
            print(f"  [降级] 主模型异常: {e}")
            return fallback_call(state)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_with_fallback)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    graph = builder.compile()
    result = graph.invoke({"messages": [HumanMessage(content="你好，请介绍一下自己")]})

    print(f"\n--- 结果 ---")
    print(f"  调用顺序: {call_log}")
    print(f"  最终结果: {result['messages'][-1].content[:200]}")


# ======================================================================
# 7. 演示五：LCEL 链异常传播与捕获
# ======================================================================
def demo_lcel_chain_error():
    """
    LCEL 管道中的异常处理：
    - 链中某一步抛异常，整个链中断
    - 用 RunnableLambda 包装 try-except 实现局部容错
    - with_fallbacks 实现链级降级
    """
    print("\n" + "=" * 70)
    print("演示五：LCEL 链异常传播与捕获")
    print("=" * 70)

    model = model_factory().create_model()

    # 1. 正常链
    print("\n--- 1. 正常链调用 ---")
    chain = model | (lambda x: x.content)
    try:
        result = chain.invoke("你好")
        print(f"  结果: {result[:100]}")
    except Exception as e:
        print(f"  异常: {e}")

    # 2. 带错误的链（模拟解析失败）
    print("\n--- 2. 链中某步抛异常 ---")

    def risky_parser(x):
        if hasattr(x, "content"):
            import json
            return json.loads(x.content)
        raise ValueError("无法解析")

    error_chain = model | RunnableLambda(risky_parser)
    try:
        result = error_chain.invoke("请用自然语言介绍自己")
        print(f"  结果: {result}")
    except Exception as e:
        print(f"  捕获异常: {type(e).__name__}: {e}")

    # 3. 用 RunnableLambda 包装容错
    print("\n--- 3. 用 RunnableLambda 包装容错 ---")

    def safe_parser(x):
        try:
            if hasattr(x, "content"):
                import json
                return json.loads(x.content)
        except Exception:
            return {"fallback": True, "raw": x.content if hasattr(x, "content") else str(x)}
        return {"fallback": True, "raw": str(x)}

    safe_chain = model | RunnableLambda(safe_parser)
    result = safe_chain.invoke("请用自然语言介绍自己")
    print(f"  结果: {result}")

    # 4. with_fallbacks 链级降级
    print("\n--- 4. with_fallbacks 链级降级 ---")

    def primary_step(x):
        raise ValueError("主链处理失败")

    def fallback_step(x):
        return f"[降级处理] 输入: {x}"

    primary = RunnableLambda(primary_step)
    fallback = RunnableLambda(fallback_step)
    safe_runnable = primary.with_fallbacks([fallback])

    result = safe_runnable.invoke("测试降级")
    print(f"  结果: {result}")


# ======================================================================
# 8. 演示六：图级别错误恢复 —— 错误计数 + 优雅退出
# ======================================================================
def demo_graph_error_recovery():
    """
    图级别错误恢复：
    - 在节点内捕获异常并记录
    - 通过状态字段追踪错误次数
    - 错误过多时优雅退出而非崩溃
    """
    print("\n" + "=" * 70)
    print("演示六：图级别错误恢复 —— 错误计数 + 优雅退出")
    print("=" * 70)

    model = model_factory().create_model()
    max_errors = 2

    def agent_node(state: AgentState) -> dict:
        """Agent 节点：带错误计数的模型调用。"""
        error_count = state.get("error_count", 0)

        if error_count >= max_errors:
            print(f"  [Agent] 错误次数已达上限 ({error_count})，优雅退出")
            return {
                "messages": [AIMessage(content=f"抱歉，处理过程中遇到了 {error_count} 次错误，暂时无法完成请求。请稍后再试。")],
            }

        try:
            response = model.invoke(state["messages"])
            return {"messages": [response], "error_count": error_count}
        except Exception as e:
            new_count = error_count + 1
            print(f"  [Agent] 模型调用失败 (第 {new_count} 次): {e}")
            return {
                "messages": [AIMessage(content="", tool_calls=[
                    {"name": "get_weather", "args": {"city": "北京"}, "id": f"err_call_{new_count}", "type": "tool_call"}
                ])],
                "error_count": new_count,
                "last_error": str(e),
            }

    def tool_node_with_guard(state: AgentState) -> dict:
        """工具节点：带防护的工具调用。"""
        messages = state["messages"]
        last_ai = messages[-1]

        tool_results = []
        for tc in (last_ai.tool_calls if hasattr(last_ai, "tool_calls") else []):
            try:
                if tc["name"] == "get_weather":
                    result = get_weather.invoke(tc["args"])
                else:
                    result = f"未知工具: {tc['name']}"
                tool_results.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
                print(f"  [工具] {tc['name']} 执行成功: {result}")
            except Exception as e:
                error_msg = f"工具 {tc['name']} 执行失败: {e}"
                tool_results.append(ToolMessage(content=error_msg, tool_call_id=tc["id"], name=tc["name"]))
                print(f"  [工具] {error_msg}")

        return {"messages": tool_results}

    def error_router(state: AgentState) -> str:
        """路由：错误过多 → END，否则继续循环。"""
        error_count = state.get("error_count", 0)
        if error_count >= max_errors:
            return END
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content and not getattr(last_msg, "tool_calls", None):
            return END
        return "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node_with_guard)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", error_router, {"agent": "agent", END: END})

    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "demo-error-recovery-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="查一下北京天气")]},
        config=config,
    )

    print(f"\n--- 结果 ---")
    print(f"  错误计数: {result.get('error_count', 0)}")
    print(f"  最后消息: {result['messages'][-1].content[:200] if result['messages'] else 'N/A'}")


# ======================================================================
# 9. 演示七：超时处理
# ======================================================================
def demo_timeout_handling():
    """
    超时处理：
    - RunnableConfig 中设置 timeout
    - 超时后自动抛异常，可配合 fallback 降级
    """
    print("\n" + "=" * 70)
    print("演示七：超时处理")
    print("=" * 70)

    def slow_operation(state: AgentState) -> dict:
        """模拟耗时操作。"""
        print(f"  [操作] 开始执行（预计 10 秒）...")
        time.sleep(10)
        return {"messages": [AIMessage(content="操作完成")]}

    def fast_fallback(state: AgentState) -> dict:
        """快速降级方案。"""
        print(f"  [降级] 快速降级方案执行...")
        return {"messages": [AIMessage(content="操作超时，已使用缓存数据返回结果。")]}

    # 方式一：在 invoke 时设置 timeout
    print("\n--- 1. invoke 级别超时 ---")
    builder = StateGraph(AgentState)
    builder.add_node("slow", slow_operation)
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)
    graph = builder.compile()

    try:
        result = graph.invoke(
            {"messages": []},
            config={"recursion_limit": 10},
        )
        print(f"  结果: {result['messages'][-1].content}")
    except Exception as e:
        print(f"  超时异常: {type(e).__name__}: {e}")

    # 方式二：带 fallback 的超时处理
    print("\n--- 2. 超时 + fallback 降级 ---")

    def with_timeout_fallback(state: AgentState) -> dict:
        try:
            return slow_operation(state)
        except Exception as e:
            print(f"  [降级] 捕获异常: {e}")
            return fast_fallback(state)

    builder2 = StateGraph(AgentState)
    builder2.add_node("safe_op", with_timeout_fallback)
    builder2.add_edge(START, "safe_op")
    builder2.add_edge("safe_op", END)
    graph2 = builder2.compile()

    result = graph2.invoke({"messages": []})
    print(f"  结果: {result['messages'][-1].content}")


# ======================================================================
# 主入口
# ======================================================================
def main():
    print("LangChain/LangGraph 异常处理完整示例")
    print("覆盖 6 类异常场景：工具异常 / 重试 / 降级 / LCEL链 / 图级恢复 / 超时\n")

    demo_tool_error_handling()
    demo_retry_policy()
    demo_custom_retry_condition()
    demo_fallback_model()
    demo_lcel_chain_error()
    demo_graph_error_recovery()
    demo_timeout_handling()

    print("\n" + "=" * 70)
    print("所有演示执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()



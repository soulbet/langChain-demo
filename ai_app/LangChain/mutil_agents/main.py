#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/9/2 12:22
@Author : zyf
@File : main.py
@Project : langChain-demo
@Software : PyCharm
@explain : 多智能体协作系统
@DESCRIPTION :
    基于 LangGraph 实现 Supervisor 模式的多智能体协作：
    - Supervisor（主管）：分析用户任务，分派给合适的专家 Agent，汇总最终结果
    - Researcher（研究员）：擅长信息检索和总结
    - Coder（程序员）：擅长编写和审查代码
    - Writer（写手）：擅长文案撰写和内容创作

    协作流程：
    用户请求 → Supervisor 分析 → 分派给专家 → 专家执行 → Supervisor 汇总 → 返回结果

    依赖：langgraph>=1.2.7, langchain>=1.3.11, langchain-core>=1.4.8
"""

import json
from typing import Annotated, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from ai_app.model_factory.model_factory import model_factory


# ======================================================================
# 1. 定义各专家 Agent 的工具
# ======================================================================

# ----- Researcher 的工具 -----
@tool
def web_search(query: str) -> str:
    """在互联网上搜索信息。"""
    return f"搜索 '{query}' 的结果：找到了 5 篇相关文章，核心内容是：LangGraph 是 LangChain 推出的有状态多 Actor 应用框架，支持循环、记忆和人工干预。"


@tool
def query_knowledge_base(topic: str) -> str:
    """从内部知识库中查询信息。"""
    return f"知识库中关于 '{topic}' 的记录：共有 12 条相关文档，最新更新时间为 2026-08-30。"


# ----- Coder 的工具 -----
@tool
def run_code(code: str, language: str = "python") -> str:
    """执行代码并返回结果。"""
    return f"代码执行成功（{language}），输出: Hello World! 运行耗时 0.03s。"


@tool
def read_file(path: str) -> str:
    """读取文件内容。"""
    return f"文件 {path} 的内容：\n```python\nprint('Hello from {path}')\n```"


# ----- Writer 的工具 -----
@tool
def save_document(title: str, content: str, format: str = "md") -> str:
    """将文档保存到指定路径。"""
    return f"文档 '{title}' 已保存为 {format} 格式，共 {len(content)} 个字符。"


@tool
def translate(text: str, target_lang: str = "en") -> str:
    """将文本翻译为目标语言。"""
    return f"翻译结果（{target_lang}）: This is the translated version of the input text."


# ======================================================================
# 2. 定义共享状态
# ======================================================================
class MultiAgentState(MessagesState):
    messages: Annotated[List, list]
    next_agent: str = ""


# ======================================================================
# 3. 构建各专家 Agent（子图）
# ======================================================================
def build_specialist_agent(
    name: str,
    system_prompt: str,
    tools: list,
    model=None,
):
    """构建一个带工具的专家 Agent 子图。"""
    if model is None:
        model = model_factory().create_model()

    llm_with_tools = model.bind_tools(tools)

    def call_model(state: MultiAgentState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MultiAgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile()


# ======================================================================
# 4. 构建 Supervisor（主管）节点
# ======================================================================
SUPERVISOR_SYSTEM_PROMPT = """你是一个智能任务协调器（Supervisor）。你的职责是：
1. 分析用户的请求
2. 决定应该由哪个专家来处理：researcher（研究员）、coder（程序员）、writer（写手）
3. 如果任务需要多个专家协作，按顺序分派
4. 汇总专家的结果，给出最终回复

可用的专家：
- researcher: 擅长信息检索、知识查询、数据分析
- coder: 擅长编写代码、审查代码、调试程序
- writer: 擅长撰写文案、翻译内容、文档编写

你必须以 JSON 格式回复，格式如下：
{"next": "专家名称 或 FINISH", "instruction": "给专家的指令 或 最终回复"}

如果任务已完成或不需要专家帮助，回复 {"next": "FINISH", "instruction": "最终回复内容"}
"""


def build_supervisor_node(model=None):
    """构建 Supervisor 节点函数。"""
    if model is None:
        model = model_factory().create_model()

    def supervisor(state: MultiAgentState) -> dict:
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        response = model.invoke(messages)

        try:
            decision = json.loads(response.content)
        except json.JSONDecodeError:
            decision = {"next": "FINISH", "instruction": response.content}

        next_agent = decision.get("next", "FINISH")
        instruction = decision.get("instruction", "")

        print(f"\n  [Supervisor] 决策 → next: {next_agent}, instruction: {instruction[:80]}...")

        return {
            "messages": [AIMessage(content=f"[Supervisor 分派给 {next_agent}]: {instruction}")],
            "next_agent": next_agent,
        }

    return supervisor


# ======================================================================
# 5. 构建路由函数
# ======================================================================
def supervisor_router(state: MultiAgentState) -> str:
    """根据 Supervisor 的决策路由到下一个节点。"""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


# ======================================================================
# 6. 专家执行包装
# ======================================================================
def _run_specialist(agent, state: MultiAgentState, name: str) -> dict:
    """调用专家 Agent 子图，并将结果包装回主图状态。"""
    print(f"\n  [{name}] 开始处理任务...")

    result = agent.invoke({"messages": state["messages"]})

    specialist_response = result["messages"][-1]
    print(f"  [{name}] 处理完成: {specialist_response.content[:80]}...")

    return {
        "messages": [AIMessage(content=f"[{name} 回复]: {specialist_response.content}")],
    }


# ======================================================================
# 7. 构建完整的多智能体图
# ======================================================================
def build_multi_agent_graph(model=None):
    """
    构建 Supervisor 模式的多智能体图：

                    ┌─────────────┐
                    │  Supervisor  │
                    └──────┬──────┘
                           │ 路由
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Researcher│ │  Coder   │ │  Writer  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           ▼
                    ┌─────────────┐
                    │  Supervisor  │  （汇总 → FINISH 或继续分派）
                    └─────────────┘
    """
    if model is None:
        model = model_factory().create_model()

    researcher = build_specialist_agent(
        name="researcher",
        system_prompt="你是一名专业的研究员，擅长信息检索和知识分析。请用简洁清晰的方式回答问题。",
        tools=[web_search, query_knowledge_base],
        model=model,
    )
    coder = build_specialist_agent(
        name="coder",
        system_prompt="你是一名资深程序员，擅长编写高质量代码。请用清晰的代码和注释回答问题。",
        tools=[run_code, read_file],
        model=model,
    )
    writer = build_specialist_agent(
        name="writer",
        system_prompt="你是一名专业写手，擅长撰写各类文案和内容。请用优美的文字回答问题。",
        tools=[save_document, translate],
        model=model,
    )

    supervisor_node = build_supervisor_node(model)

    builder = StateGraph(MultiAgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", lambda state: _run_specialist(researcher, state, "Researcher"))
    builder.add_node("coder", lambda state: _run_specialist(coder, state, "Coder"))
    builder.add_node("writer", lambda state: _run_specialist(writer, state, "Writer"))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {"researcher": "researcher", "coder": "coder", "writer": "writer", END: END},
    )

    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("writer", "supervisor")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ======================================================================
# 8. 演示
# ======================================================================
def demo_research_task():
    """演示：研究类任务 → 分派给 Researcher"""
    print("\n" + "=" * 70)
    print("演示一：研究类任务")
    print("=" * 70)

    graph = build_multi_agent_graph()
    config = {"configurable": {"thread_id": "demo-research-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我查一下 LangGraph 是什么？它有哪些核心特性？")]},
        config=config,
    )

    print("\n--- 最终结果 ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"  {msg.content[:200]}")


def demo_code_task():
    """演示：编程类任务 → 分派给 Coder"""
    print("\n" + "=" * 70)
    print("演示二：编程类任务")
    print("=" * 70)

    graph = build_multi_agent_graph()
    config = {"configurable": {"thread_id": "demo-code-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我写一个 Python 快速排序算法，并运行测试一下。")]},
        config=config,
    )

    print("\n--- 最终结果 ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"  {msg.content[:200]}")


def demo_write_task():
    """演示：写作类任务 → 分派给 Writer"""
    print("\n" + "=" * 70)
    print("演示三：写作类任务")
    print("=" * 70)

    graph = build_multi_agent_graph()
    config = {"configurable": {"thread_id": "demo-write-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我写一段关于人工智能未来发展的短文，并翻译成英文。")]},
        config=config,
    )

    print("\n--- 最终结果 ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"  {msg.content[:200]}")


def demo_multi_step_task():
    """演示：多步协作任务 → Supervisor 多次分派"""
    print("\n" + "=" * 70)
    print("演示四：多步协作任务（Supervisor 多次分派）")
    print("=" * 70)

    graph = build_multi_agent_graph()
    config = {"configurable": {"thread_id": "demo-multi-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(
            content="我想做一个项目：1. 先调研一下 Python Web 框架的最新趋势 "
                    "2. 然后写一个 FastAPI 的示例代码 "
                    "3. 最后写一份项目说明文档"
        )]},
        config=config,
    )

    print("\n--- 最终结果 ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"  {msg.content[:200]}")


# ======================================================================
# 主入口
# ======================================================================
def main():
    print("多智能体协作系统（Supervisor 模式）")
    print("架构：Supervisor ←→ Researcher / Coder / Writer")
    print("说明：Supervisor 分析任务 → 分派专家 → 汇总结果\n")

    demo_research_task()
    demo_code_task()
    demo_write_task()
    demo_multi_step_task()

    print("\n" + "=" * 70)
    print("所有演示执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()

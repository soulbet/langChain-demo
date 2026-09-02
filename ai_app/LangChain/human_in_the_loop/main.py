#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/9/2 11:19
@Author : zyf
@File : main.py
@Project : langChain-demo
@Software : PyCharm
@explain : Human-in-the-Loop 完整示例
@DESCRIPTION :
    演示 LangGraph 中 Human-in-the-Loop 的三种核心模式：
    1. interrupt_before —— 在工具节点执行前中断，等待人工审批
    2. interrupt_after  —— 在工具节点执行后中断，等待人工确认结果
    3. interrupt() 函数  —— 在工具内部主动中断，支持批准/修改参数/拒绝三种决策

    依赖：langgraph>=1.2.7, langchain>=1.3.11, langchain-core>=1.4.8
"""

from typing import Annotated, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


from ai_app.model_factory.model_factory import model_factory


# ======================================================================
# 1. 定义工具
# ======================================================================
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息。"""
    return f"{city} 天气晴朗，气温 25°C，湿度 60%。"


@tool
def delete_file(path: str) -> str:
    """删除指定路径的文件（高风险操作）。"""
    return f"已成功删除文件: {path}"


@tool
def send_email(to: str, subject: str, content: str) -> str:
    """发送邮件给指定用户（敏感操作）。"""
    return f"已发送邮件至 {to}，主题: {subject}"


@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息，工具内部使用 interrupt() 主动暂停以获取人工确认。"""
    human_decision = interrupt({
        "question": f"即将在数据库中执行搜索: '{query}'，是否继续？",
        "options": ["approve", "modify", "reject"],
    })
    if human_decision.get("decision") == "approve":
        return f"搜索 '{query}' 的结果: 找到 3 条相关记录。"
    elif human_decision.get("decision") == "modify":
        new_query = human_decision.get("new_query", query)
        return f"搜索 '{new_query}' 的结果: 找到 5 条相关记录。"
    else:
        return "搜索已被用户取消。"


# ======================================================================
# 2. 定义状态
# ======================================================================
class AgentState(MessagesState):
    # 可以加任意多个自定义字段

    ## List 类型，表示消息列表
    ## list：reducer 函数（这里是 list，表示新消息追加到末尾）
    ## 自定义reducer：Annotated[int, lambda old, new: old + new]
    messages: Annotated[List, list] # 消息列表


# ======================================================================
# 3. 构建 Agent 图
# ======================================================================
def build_agent_graph(tools: list, model=None):
    """构建一个带工具调用能力的 Agent 图。"""
    if model is None:
        model = model_factory().create_model()

    llm_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    # tools_condition 的源码在 langgraph/prebuilt/tool_node.py 第 1582-1659 行。
    # 判断‘agent’节点是否返回工具调用，
    # "tools": "tools" key:tools_condition 函数的返回值（路由条件),value:条件匹配后要跳转到的目标节点名
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder


# ======================================================================
# 4. 演示一：interrupt_before —— 工具执行前中断（人工审批）
# ======================================================================
def demo_interrupt_before():
    """
    interrupt_before 会在指定节点执行前暂停图运行。
    典型场景：高风险工具（删除文件、发送邮件）需要先让人工审批。
    """
    print("\n" + "=" * 70)
    print("演示一：interrupt_before —— 工具执行前中断（人工审批）")
    print("=" * 70)

    tools = [get_weather, delete_file, send_email]
    builder = build_agent_graph(tools)

    checkpointer = MemorySaver()
    # 把图的定义变成可执行的
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )

    config = {"configurable": {"thread_id": "demo-before-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我删除 /tmp/old_data.txt 文件，然后查一下北京天气。")]},
        config=config,
    )

    print("\n--- 图已中断，当前待执行的工具调用 ---")
    last_ai = result["messages"][-1]
    if hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
        for tc in last_ai.tool_calls:
            print(f"  工具: {tc['name']}, 参数: {tc['args']}")

    print("\n--- 人工审批：批准所有操作，恢复执行 ---")
    result = graph.invoke(Command(resume=None), config=config)

    print("\n--- 最终结果 ---")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  [{msg.name}] {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"  [AI] {msg.content}")


# ======================================================================
# 5. 演示二：interrupt_after —— 工具执行后中断（人工确认结果）
# ======================================================================
def demo_interrupt_after():
    """
    interrupt_after 会在指定节点执行完成后暂停图运行。
    典型场景：工具已执行，但需要人工确认结果后再让 Agent 继续。
    """
    print("\n" + "=" * 70)
    print("演示二：interrupt_after —— 工具执行后中断（人工确认结果）")
    print("=" * 70)

    tools = [get_weather, delete_file]
    builder = build_agent_graph(tools)

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["tools"],
    )

    config = {"configurable": {"thread_id": "demo-after-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="查一下上海天气怎么样？")]},
        config=config,
    )

    print("\n--- 工具已执行完毕，图已中断，查看工具返回结果 ---")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  [{msg.name}] {msg.content}")

    print("\n--- 人工确认结果无误，恢复执行 ---")
    result = graph.invoke(Command(resume=None), config=config)

    print("\n--- 最终 AI 回复 ---")
    last_msg = result["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.content:
        print(f"  [AI] {last_msg.content}")


# ======================================================================
# 6. 演示三：interrupt() 函数 —— 工具内部主动中断（动态决策）
# ======================================================================
def demo_interrupt_function():
    """
    在工具内部调用 interrupt() 主动暂停图执行。
    人工可以通过 Command(resume={...}) 传回决策：
      - approve: 批准执行
      - modify: 修改参数后继续
      - reject: 拒绝执行
    """
    print("\n" + "=" * 70)
    print("演示三：interrupt() 函数 —— 工具内部主动中断（动态决策）")
    print("=" * 70)

    tools = [search_database, get_weather]
    builder = build_agent_graph(tools)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "demo-func-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我在数据库中搜索 'LangChain 教程'")]},
        config=config,
    )

    print("\n--- 工具内部已触发 interrupt，等待人工决策 ---")
    print(f"  中断信息: {result.get('__interrupt__')}")

    print("\n--- 人工决策：批准搜索 ---")
    result = graph.invoke(
        Command(resume={"decision": "approve"}),
        config=config,
    )

    print("\n--- 最终结果 ---")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  [{msg.name}] {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"  [AI] {msg.content}")


# ======================================================================
# 7. 演示四：多轮审批 —— 模拟真实场景中的逐步确认
# ======================================================================
def demo_multi_step_approval():
    """
    模拟真实业务场景：Agent 要执行多个敏感操作，
    人工逐一审批每个操作（approve / reject）。
    """
    print("\n" + "=" * 70)
    print("演示四：多轮审批 —— 模拟真实场景中的逐步确认")
    print("=" * 70)

    tools = [delete_file, send_email, get_weather]
    builder = build_agent_graph(tools)

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )

    config = {"configurable": {"thread_id": "demo-multi-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(
            content="请帮我：1. 删除 /data/report.csv  2. 给 boss@company.com 发一封主题为'周报'的邮件  3. 查一下深圳天气"
        )]},
        config=config,
    )

    print("\n--- 第一轮中断：查看待执行的工具 ---")
    last_ai = result["messages"][-1]
    if hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
        for i, tc in enumerate(last_ai.tool_calls):
            print(f"  [{i+1}] 工具: {tc['name']}, 参数: {tc['args']}")

    print("\n--- 人工审批：只批准天气查询，拒绝其他操作 ---")
    print("  (实际应用中，可以通过修改 tool_calls 来实现部分批准)")
    result = graph.invoke(Command(resume=None), config=config)

    print("\n--- 执行完毕，最终消息 ---")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  [{msg.name}] {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"  [AI] {msg.content}")


# ======================================================================
# 主入口
# ======================================================================
def main():
    print("LangGraph Human-in-the-Loop 完整演示")
    print("说明：本示例演示了 4 种 HITL 模式，依次运行...\n")

    demo_interrupt_before()
    demo_interrupt_after()
    demo_interrupt_function()
    demo_multi_step_approval()

    print("\n" + "=" * 70)
    print("所有演示执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/9/2 11:19
@Author : zyf
@File : main.py
@Project : langChain-demo
@Software : PyCharm
@explain :
@DESCRIPTION :
"""

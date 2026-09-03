#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/9/2 17:05
@Author : zyf
@File : main.py
@Project : langChain-demo
@Software : PyCharm
@explain : 长短期记忆完整示例
@DESCRIPTION :
    基于 LangGraph 实现 Agent 的长短期记忆系统，突出 4 个应用场景：

    【短期记忆】MemorySaver（checkpointer）
    - 作用域：单次会话（同一 thread_id 内）
    - 生命周期：会话结束即丢失
    - 场景：多轮对话上下文、工具调用状态恢复

    【长期记忆】InMemoryStore（store）
    - 作用域：跨会话（不同 thread_id 之间共享）
    - 生命周期：进程存活期间永久（生产用 PostgresStore）
    - 场景：用户画像、偏好设置、知识积累

    应用场景：
    1. 智能客服 —— 短期记住当前工单上下文，长期记住用户偏好和历史
    2. 私人助理 —— 短期记住当天对话，长期记住用户习惯和重要日期
    3. 教育辅导 —— 短期记住当前题目上下文，长期记住学生薄弱点和学习进度
    4. 记忆管理 —— 查看/更新/删除长期记忆

    依赖：langgraph>=1.2.7, langchain>=1.3.11, langchain-core>=1.4.8
"""

from typing import Annotated, List, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore




# ======================================================================
# 1. 定义工具 —— 通过 InjectedStore 访问长期记忆
# ======================================================================
from typing_extensions import Annotated as TAnnotated
from langgraph.prebuilt import InjectedStore

from ai_app.model_factory.model_factory import ModelFactory


@tool
def save_user_preference(key: str, value: str, store: TAnnotated[Any, InjectedStore()]) -> str:
    """保存用户偏好到长期记忆。例如：保存喜欢的语言为Python。"""
    store.put(("user", "preferences"), key, {"value": value})
    return f"已记住：{key} = {value}"


@tool
def get_user_preference(key: str, store: TAnnotated[Any, InjectedStore()]) -> str:
    """从长期记忆中查询用户偏好。"""
    item = store.get(("user", "preferences"), key)
    if item:
        return f"记忆中 {key} = {item.value.get('value', '未知')}"
    return f"记忆中没有关于 '{key}' 的记录"


@tool
def save_user_fact(fact_key: str, fact: str, store: TAnnotated[Any, InjectedStore()]) -> str:
    """保存用户的重要事实到长期记忆。例如：用户的生日、公司名称等。"""
    store.put(("user", "facts"), fact_key, {"fact": fact})
    return f"已记住事实：{fact_key} → {fact}"


@tool
def get_user_fact(fact_key: str, store: TAnnotated[Any, InjectedStore()]) -> str:
    """从长期记忆中查询用户事实。"""
    item = store.get(("user", "facts"), fact_key)
    if item:
        return f"记忆中 {fact_key} → {item.value.get('fact', '未知')}"
    return f"记忆中没有关于 '{fact_key}' 的事实"


@tool
def list_memories(namespace: str = "preferences", store: TAnnotated[Any, InjectedStore()] = None) -> str:
    """列出指定命名空间下的所有记忆。namespace 可选值: preferences, facts, learning。"""
    items = store.list_items(("user", namespace))
    if not items:
        return f"命名空间 '{namespace}' 下没有记忆"
    lines = [f"  - {item.key}: {item.value}" for item in items]
    return f"记忆列表（{namespace}）:\n" + "\n".join(lines)


@tool
def delete_memory(key: str, namespace: str = "preferences", store: TAnnotated[Any, InjectedStore()] = None) -> str:
    """删除指定的长期记忆。"""
    store.delete(("user", namespace), key)
    return f"已删除记忆: ({namespace}) {key}"


@tool
def save_learning_progress(topic: str, status: str, store: TAnnotated[Any, InjectedStore()]) -> str:
    """保存学习进度。status 可选: started, in_progress, mastered, difficult。"""
    store.put(("user", "learning"), topic, {"status": status})
    return f"学习进度已更新：{topic} → {status}"


@tool
def get_learning_progress(topic: str, store: TAnnotated[Any, InjectedStore()]) -> str:
    """查询某知识点的学习进度。"""
    item = store.get(("user", "learning"), topic)
    if item:
        return f"{topic} 的学习状态: {item.value.get('status', '未知')}"
    return f"没有 {topic} 的学习记录"


@tool
def get_weather(city: str) -> str:
    """获取城市天气信息。"""
    return f"{city} 天气晴朗，气温 25°C，湿度 60%。"


# ======================================================================
# 2. 定义状态
# ======================================================================
class AgentState(MessagesState):
    messages: Annotated[List, list]


# ======================================================================
# 3. 构建带长短期记忆的 Agent 图
# ======================================================================
def build_memory_agent(model=None):
    """
    构建同时具备短期记忆（checkpointer）和长期记忆（store）的 Agent。

    短期记忆 = MemorySaver + thread_id
      → 同一 thread_id 内的多轮对话自动保持上下文

    长期记忆 = InMemoryStore + InjectedStore
      → 跨 thread_id 共享，工具通过 store.put/get 读写

    架构：
    用户消息 → [Agent（读长期记忆注入 system prompt）]
                    ⇄ [Tools（读写长期记忆）]
               → 返回回复（短期记忆自动保存）
    """
    if model is None:
        model = ModelFactory().create_model()

    tools = [
        save_user_preference, get_user_preference,
        save_user_fact, get_user_fact,
        list_memories, delete_memory,
        save_learning_progress, get_learning_progress,
        get_weather,
    ]
    llm_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    # 同时配置 checkpointer（短期记忆）和 store（长期记忆）
    checkpointer = MemorySaver()
    store = InMemoryStore()

    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph, store


# ======================================================================
# 4. 场景一：智能客服 —— 短期记住工单上下文，长期记住用户偏好
# ======================================================================
def demo_customer_service():
    """
    场景：智能客服系统
    - 短期记忆：记住当前工单的对话上下文（用户描述的问题、排查步骤）
    - 长期记忆：记住用户的偏好（语言偏好、沟通风格、历史问题）
    """
    print("\n" + "=" * 70)
    print("场景一：智能客服 —— 短期工单上下文 + 长期用户偏好")
    print("=" * 70)

    graph, store = build_memory_agent()

    # ===== 第一次会话：用户设置偏好 =====
    print("\n--- 第一次会话：用户告知偏好 ---")
    config1 = {"configurable": {"thread_id": "ticket-001"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我叫张三，我喜欢用中文交流，请用简洁的方式回复我。")]},
        config=config1,
    )
    print(f"  客服: {result['messages'][-1].content[:150]}")

    # 模拟工具保存偏好
    store.put(("user", "preferences"), "language", {"value": "中文"})
    store.put(("user", "preferences"), "style", {"value": "简洁"})
    store.put(("user", "preferences"), "name", {"value": "张三"})
    print("  [系统] 已保存用户偏好到长期记忆")

    # ===== 第二次会话：用户换新工单，但 Agent 仍记得偏好 =====
    print("\n--- 第二次会话：新工单，但长期记忆仍在 ---")
    config2 = {"configurable": {"thread_id": "ticket-002"}}

    # 手动将长期记忆注入 system prompt
    prefs = store.list_items(("user", "preferences"))
    memory_context = "用户历史偏好：" + "; ".join(
        [f"{p.key}={p.value.get('value', '')}" for p in prefs]
    )
    print(f"  [系统] 从长期记忆加载: {memory_context}")

    result = graph.invoke(
        {"messages": [
            SystemMessage(content=f"你是智能客服。{memory_context}。请根据用户偏好回复。"),
            HumanMessage(content="我的系统又报错了，帮我看看。"),
        ]},
        config=config2,
    )
    print(f"  客服: {result['messages'][-1].content[:150]}")

    # ===== 第三次调用：同一工单内多轮对话（短期记忆） =====
    print("\n--- 同一工单内多轮对话（短期记忆生效）---")
    result = graph.invoke(
        {"messages": [HumanMessage(content="刚才说的那个错误，具体是报 NullPointerException")]},
        config=config2,
    )
    print(f"  客服: {result['messages'][-1].content[:150]}")


# ======================================================================
# 5. 场景二：私人助理 —— 短期记住当天对话，长期记住重要事实
# ======================================================================
def demo_personal_assistant():
    """
    场景：私人助理
    - 短期记忆：记住今天的对话内容（约饭、安排日程）
    - 长期记忆：记住用户的重要事实（生日、家庭成员、公司地址）
    """
    print("\n" + "=" * 70)
    print("场景二：私人助理 —— 短期当天对话 + 长期重要事实")
    print("=" * 70)

    graph, store = build_memory_agent()

    # ===== 预先存储长期记忆 =====
    store.put(("user", "facts"), "birthday", {"fact": "1990年5月15日"})
    store.put(("user", "facts"), "spouse", {"fact": "妻子叫李梅"})
    store.put(("user", "facts"), "company", {"fact": "在字节跳动工作"})
    store.put(("user", "facts"), "allergy", {"fact": "对花生过敏"})
    print("\n--- 预存长期记忆（用户重要事实）---")
    for key in ["birthday", "spouse", "company", "allergy"]:
        item = store.get(("user", "facts"), key)
        print(f"  {key}: {item.value.get('fact', '')}")

    # ===== 当天对话（短期记忆） =====
    print("\n--- 当天对话（短期记忆保持上下文）---")
    config = {"configurable": {"thread_id": "assistant-today"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我订一个餐厅，今晚和妻子一起吃饭。")]},
        config=config,
    )
    print(f"  助理: {result['messages'][-1].content[:150]}")

    # 追问（短期记忆知道"妻子"是谁）
    result = graph.invoke(
        {"messages": [HumanMessage(content="对，就是我跟你说过的我妻子，她不吃花生。")]},
        config=config,
    )
    print(f"  助理: {result['messages'][-1].content[:150]}")

    # ===== 第二天新对话（短期记忆清空，长期记忆仍在） =====
    print("\n--- 第二天新对话（短期记忆已清空，长期记忆仍在）---")
    config2 = {"configurable": {"thread_id": "assistant-tomorrow"}}

    facts = store.list_items(("user", "facts"))
    facts_context = "用户信息：" + "; ".join(
        [f"{f.key}={f.value.get('fact', '')}" for f in facts]
    )

    result = graph.invoke(
        {"messages": [
            SystemMessage(content=f"你是私人助理。{facts_context}。请根据用户信息提供个性化服务。"),
            HumanMessage(content="下周我妻子生日，帮我准备个惊喜。"),
        ]},
        config=config2,
    )
    print(f"  助理: {result['messages'][-1].content[:200]}")


# ======================================================================
# 6. 场景三：教育辅导 —— 短期记住题目上下文，长期记住学习进度
# ======================================================================
def demo_education_tutor():
    """
    场景：AI 教育辅导
    - 短期记忆：记住当前题目的推导过程
    - 长期记忆：记住学生的薄弱知识点和学习进度
    """
    print("\n" + "=" * 70)
    print("场景三：教育辅导 —— 短期题目上下文 + 长期学习进度")
    print("=" * 70)

    graph, store = build_memory_agent()

    # ===== 预存学习进度 =====
    store.put(("user", "learning"), "python_basic", {"status": "mastered"})
    store.put(("user", "learning"), "python_oop", {"status": "in_progress"})
    store.put(("user", "learning"), "python_decorator", {"status": "difficult"})
    store.put(("user", "learning"), "sql_join", {"status": "mastered"})
    print("\n--- 学生历史学习进度（长期记忆）---")
    for topic in ["python_basic", "python_oop", "python_decorator", "sql_join"]:
        item = store.get(("user", "learning"), topic)
        print(f"  {topic}: {item.value.get('status', '')}")

    # ===== 当前辅导对话（短期记忆） =====
    print("\n--- 辅导对话：当前题目上下文（短期记忆）---")
    config = {"configurable": {"thread_id": "tutor-session-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="请教我 Python 装饰器，我总是搞不懂。")]},
        config=config,
    )
    print(f"  老师: {result['messages'][-1].content[:150]}")

    # 追问（短期记忆知道在讲装饰器）
    result = graph.invoke(
        {"messages": [HumanMessage(content="那 @staticmethod 和 @classmethod 有什么区别？")]},
        config=config,
    )
    print(f"  老师: {result['messages'][-1].content[:150]}")

    # 更新学习进度
    store.put(("user", "learning"), "python_decorator", {"status": "in_progress"})
    store.put(("user", "learning"), "python_classmethod", {"status": "started"})
    print("\n--- 更新学习进度（写入长期记忆）---")
    for topic in ["python_decorator", "python_classmethod"]:
        item = store.get(("user", "learning"), topic)
        print(f"  {topic}: {item.value.get('status', '')}")


# ======================================================================
# 7. 场景四：记忆管理 —— 查看/更新/删除
# ======================================================================
def demo_memory_management():
    """
    场景：记忆管理
    展示如何对长期记忆进行 CRUD 操作。
    """
    print("\n" + "=" * 70)
    print("场景四：记忆管理 —— 查看/更新/删除长期记忆")
    print("=" * 70)

    _, store = build_memory_agent()

    # ===== Create 创建 =====
    print("\n--- Create：写入记忆 ---")
    store.put(("user", "preferences"), "theme", {"value": "dark"})
    store.put(("user", "preferences"), "language", {"value": "中文"})
    store.put(("user", "facts"), "job", {"fact": "Python 开发工程师"})
    print("  已写入 3 条记忆")

    # ===== Read 读取 =====
    print("\n--- Read：读取记忆 ---")
    item = store.get(("user", "preferences"), "theme")
    print(f"  get('theme') = {item.value}")

    # ===== List 列出 =====
    print("\n--- List：列出命名空间下所有记忆 ---")
    items = store.list_items(("user", "preferences"))
    for item in items:
        print(f"  {item.key}: {item.value}")

    # ===== Update 更新 =====
    print("\n--- Update：更新记忆 ---")
    store.put(("user", "preferences"), "theme", {"value": "light"})
    item = store.get(("user", "preferences"), "theme")
    print(f"  更新后 theme = {item.value}")

    # ===== Delete 删除 =====
    print("\n--- Delete：删除记忆 ---")
    store.delete(("user", "preferences"), "language")
    items = store.list_items(("user", "preferences"))
    print(f"  删除 language 后，剩余记忆:")
    for item in items:
        print(f"    {item.key}: {item.value}")

    # ===== 跨命名空间查看 =====
    print("\n--- 跨命名空间查看 ---")
    for ns in ["preferences", "facts", "learning"]:
        items = store.list_items(("user", ns))
        print(f"  ({ns}): {len(items)} 条记忆")


# ======================================================================
# 主入口
# ======================================================================
def main():
    print("长短期记忆系统完整示例")
    print("短期记忆 = MemorySaver（checkpointer）→ 单次会话内上下文")
    print("长期记忆 = InMemoryStore（store）→ 跨会话永久存储")
    print("4 个应用场景：智能客服 / 私人助理 / 教育辅导 / 记忆管理\n")

    demo_customer_service()
    demo_personal_assistant()
    demo_education_tutor()
    demo_memory_management()

    print("\n" + "=" * 70)
    print("所有演示执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()

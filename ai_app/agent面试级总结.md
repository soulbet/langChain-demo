# 🤖 Agent 面试级总结（含关键词 / 知识点 / 场景）

> 目录：`ai_app`
> 编写依据：本项目 `ai_app` 下的真实实现（LangChain / LangGraph / DeepAgents / MCP / RAG / ModelFactory）
> 目标：把「AI Agent」这一知识体系做成一份可背、可讲、可答的面试手册。

---

## 目录

1. [Agent 是什么 —— 一句话定位](#1-agent-是什么--一句话定位)
2. [核心关键词表（面试高频词）](#2-核心关键词表面试高频词)
3. [Agent 内核：四大经典范式](#3-agent-内核四大经典范式)
4. [工具调用 Tool / Function Calling](#4-工具调用-tool--function-calling)
5. [记忆 Memory](#5-记忆-memory)
6. [RAG + Agent（本项目落地）](#6-rag--agent本项目落地)
7. [多 Agent 编排 / Multi-Agent](#7-多-agent-编排--multi-agent)
8. [状态机与持久化：LangGraph](#8-状态机与持久化langgraph)
9. [中间件与横切关注点](#9-中间件与横切关注点)
10. [执行环境 / 沙箱（DeepAgents）](#10-执行环境--沙箱deepagents)
11. [模型工厂 / 多模型路由（ModelFactory）](#11-模型工厂--多模型路由modelfactory)
12. [MCP 与工具生态](#12-mcp-与工具生态)
13. [Agent 常见架构拓扑与选型](#13-agent-常见架构拓扑与选型)
14. [高频面试题速答](#14-高频面试题速答)
15. [生产级考量：可观测 / 安全 / 成本 / 评测](#15-生产级考量可观测--安全--成本--评测)
16. [一页带走](#16-一页带走)

---

## 1. Agent 是什么 —— 一句话定位

**Agent（智能体）= LLM（大脑）+ 工具（手脚）+ 记忆（经历）+ 循环决策（工作方式）**。

- **LLM**：负责理解与推理，是"决策中枢"。
- **工具 Tool**：让 LLM 有能力**真正行动**（查库、调 API、跑代码）——这是 Agent 区别于纯 Chat 的核心。
- **记忆 Memory**：跨轮次记住上下文，让 Agent"有连续性"。
- **循环决策 循环**：Agent 不是"一答到底"，而是
  `观察(Observe) → 思考(Reason) → 行动(Act) → 再次观察...直到完成`。

> 与纯 LLM 项目的本质区别：**纯 LLM 只输出文本；Agent 会输出"动作指令"（tool_calls）并循环执行直到得到最终结果。**

---

## 2. 核心关键词表（面试高频词）

| 关键词 | 直译/别称 | 一句话说明 | 本项目位置 |
|--------|-----------|-----------|-----------|
| **ReAct** | Reason + Act | 让模型"边推理边行动"的经典范式（`Thought → Action → Observation`） | Agent 内核 |
| **Tool Calling / Function Calling** | 工具调用 / 函数调用 | 模型按 schema 输出结构化的工具调用参数 | langchain tools / `@tool` |
| **Tool Schema** | 工具描述 | 用 Pydantic/JSON Schema 描述工具的参数与用途 | `@tool` 自动生成 |
| **LCEL / Runnable** | 链式表达式 | `\|` 管道组合组件，统一 `invoke/stream/batch` | `prompt.md` |
| **StateGraph** | 状态机图 | 用节点(node)+边(edge)描述有状态、含循环的流程 | `LangGraph/` |
| **Checkpointer** | 检查点 | 保存/恢复图状态，实现跨轮记忆与中断恢复 | `MemorySaver`/`PostgresSaver` |
| **Middleware** | 中间件 | 在 agent 主循环关键节点插入逻辑（重试/回退/脱敏等） | `middleware/` |
| **Memory** | 记忆 | 短期（会话内）/长期（持久化到 DB/向量库） | langgraph checkpointer |
| **RAG** | 检索增强生成 | 检索+上下文+生成，缓解幻觉 | `rag_agents/` |
| **Embedding** | 向量化 | 把文本转成向量做语义检索 | `nomic-embed-text` |
| **Hybrid Search** | 混合检索 | 语义检索 + 关键词检索 相结合 | `向量 + BM25` |
| **RRF** | Reciprocal Rank Fusion | 多路召回排名融合算法 `score=Σ1/(k+rank)` | `tools.rrf` |
| **Reranker / 精排** | CrossEncoder | 对召回结果再排序，提高准确率 | `BGE-reranker-v2-m3` |
| **Multi-Agent** | 多智能体 | 多个 Agent 分工/协作/竞争完成复杂任务 | `agency-agents/` |
| **Supervisor** | 主管 | 一个"调度 Agent"决定把任务分给哪个子 Agent | 多 Agent 拓扑 |
| **Parallel / Sequential** | 并行/串行 | 多 Agent 同时做 / 按序做 | 编排模式 |
| **Reflection** | 反思 | Agent 自我评估/自我修正（critic→改进） | 高级范式 |
| **Plan-and-Execute** | 规划-执行 | 先拆解计划，再逐步执行 | 高级范式 |
| **Sandbox** | 沙箱 | 隔离安全的代码/命令执行环境 | `create_deep_agent` + `LocalShellBackend` |
| **HITL** | Human-in-the-loop | 关键节点人工介入确认 | `HumanInTheLoopMiddleware` |
| **MCP** | Model Context Protocol | 统一接入外部工具/数据源的开放协议 | `langchain_mcp_adapters`/`fastmcp` |
| **Prompt Injection** | 提示注入 | 恶意文本诱导 Agent 违规（安全隐患） | 安全章节 |
| **Hallucination** | 幻觉 | 模型编造不存在的事实 | RAG/评测对抗 |
| **structured output** | 结构化输出 | 让模型按 Pydantic schema 输出，强约束 | `with_structured_output` |
| **Token/Latency/Cost** | Token/时延/成本 | 三大约束指标 | 生产考量 |
| **Evaluation** | 评测 | 用基准/召回率/准确率评估效果 | RAG 评测 |

---

## 3. Agent 内核：四大经典范式

### 3.1 ReAct（最常用）
模型自由交替"思考 → 行动 → 观察"，直到给出答案。
对应 LangChain v1 的 `create_agent`（底层由 LangGraph 驱动）：

```python
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=tools, system_prompt=...)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

### 3.2 Plan-and-Execute（规划-执行）
先让模型**拆解成计划**，再一步步执行、收集结果。适合长任务、可分解的任务。

### 3.3 Reflection（反思）
模型产出结果后，再用"批判者"评审/修正，可多轮迭代。能明显提升质量，但成本更高。

### 3.4 Multi-Agent（多智能体）
多个 Agent 各司其职，通过**调度/通讯**协作。常见拓扑见第 13 节。

> 面试常问：**受控循环 vs 递归深度** —— `recursion_limit`（LangGraph 默认 25）用于防止无限循环；循环型 Agent 必须有明确的终止条件/步数上限。

---

## 4. 工具调用 Tool / Function Calling

### 4.1 定义工具
```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""          # docstring 会作为工具描述给模型
    return a + b
```
- **签名 + 类型注解 + docstring** 自动生成 tool schema，模型据此决定何时调用、传什么参数。
- 可组合返回 `ToolMessage`。

### 4.2 工具在循环中的流程
```
LLM 输出 tool_calls → ToolNode 执行工具 → ToolMessage 回填 → 再次调用 LLM → ... → LLM 输出最终答案
```
本项目 `LangGraph/persistence_demo` 里的 `ToolNode` + `tools_condition` 即典型实现。

### 4.3 关键设计问题（面试高频）
| 问题 | 要点 |
|------|------|
| 工具描述怎么写才好？ | 描述要明确"何时用、何时不用、参数含义"；越清晰命中率越高 |
| 工具太多怎么办？ | 用 `ToolSelectionMiddleware` 做工具选择；或分层/分组 |
| 工具参数错误？ | `wrap_tool_call` 里校验/修正参数、或重试 |
| 工具返回太长？ | 截断/摘要（`SummarizationMiddleware`）、只保留结构化结果 |
| 模型不支持工具？ | 用 `tool_choice` 强制 / `with_structured_output` |
| 权限/危险操作？ | 最小权限；高危动作人工确认（HITL）或沙箱 |

---

## 5. 记忆 Memory

| 维度 | 说明 | 实现 |
|------|------|------|
| **短期记忆** | 当前会话内的消息 | 直接把 messages 喂回模型 |
| **长期记忆** | 跨会话/线程持久化 | LangGraph `checkpointer`（DB），或向量库存历史摘要 |
| **上下文压缩** | 消息太多时摘要折叠 | `SummarizationMiddleware` / `TrimMessages` |
| **状态字段** | 自定义业务状态 | `state_schema` / `state` dict |

本项目示例（`LangGraph/persistence_demo`）：
```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()            # 测试用；生产换 PostgresSaver
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "user-123"}}
# 同一 thread_id 多次 invoke，Agent 自动从检查点恢复上下文
```

> 生产推荐 **PostgresSaver / RedisSaver** 做持久化，而不是内存。

---

## 6. RAG + Agent（本项目落地）

本项目在 `ai_app/main/rag_agents/` 实现了一套**生产级 RAG**，是"检索增强"的最佳示例。

### 6.1 标准流程
```
用户问题
  → Query Analysis
  → Vector Search (PGVector Top20)   +   Keyword Search (ES BM25 Top20)
  → RRF 融合 (k=60) → Hybrid Top100
  → BGE Reranker 精排 → Top3
  → LLM
  → 最终答案
```

### 6.2 关键组件
| 组件 | 作用 | 库 |
|------|------|-----|
| 文档加载 | 读文本文件 | `TextLoader`（`langchain_community`） |
| 切分 | `RecursiveCharacterTextSplitter`（chunk_size/overlap） | `langchain_text_splitters` |
| 向量化 | `nomic-embed-text` | `ModelFactory` + LLM 本地服务 |
| 语义检索 | PGVector（`vector` 列） | `langchain_postgres` + pgvector |
| 关键词检索 | Elasticsearch BM25（`ik` 分词/语义文本） | `elasticsearch` |
| 融合排序 | RRF `score += 1/(k+rank)` | 手写 `tools.rrf` |
| 精排 | BGE CrossEncoder `model.predict([(q, doc)])` | `sentence_transformers` |
| 组装 | 片段拼接成 context 喂给 LLM | `Tools.retrieve_context` |

### 6.3 RRF 与 Reranker 的分工（面试必背）
- **RRF**：负责"多路召回融合" → **提高召回率**；只按排名、不理解内容、无需训练。
- **BGE Reranker**：负责"相关性精排" → **提高准确率**；真正理解 query-文档关系，但**成本高、慢**。
- 口诀：**RRF 召回、Reranker 精排；召回提率、精排提质。**

### 6.4 RAG Agent 常见扩展
- **Query Rewriting / Query Analysis**：改写问题、拆解多跳问题、生成假答案检索。
- **Hybrid Search**：向量+关键词互补（语义强 vs 关键词强）。
- **多跳检索 / Graph RAG**：知识图谱增强，多跳推理（`Knowledge Graph Engineer`）。
- **Rerank**：已含。
- **评测**：召回率、MRR、NDCG、答案正确率、防幻觉测评。

---

## 7. 多 Agent 编排 / Multi-Agent

多 Agent 让复杂任务被"分解 + 专业化"，但引入**通讯、上下文、信任、故障恢复**问题。

### 7.1 主流拓扑
| 拓扑 | 描述 | 适用 |
|------|------|------|
| **Supervisor（主管）** | 一个调度 Agent 把任务分给子 Agent 并汇总 | 任务可分解、需分工 |
| **Hierarchical（层级）** | 主管 → 组长 → 组员，多层 | 大规模、强治理 |
| **Router（路由）** | 按意图把请求路由到不同专家 Agent | 意图清晰、互不重叠 |
| **Sequential（串行）** | 前一个输出喂给下一个 | 流水线式流程 |
| **Parallel（并行）** | 同时跑多个，最后汇总 | 可并行的子任务 |
| **Network（网络/竞争）** | Agent 互相通信/辩论 | 复杂协作、评估/生成对抗 |
| **Swarm（群体）** | 动态协商、自我组织 | 高度动态、去中心化 |

### 7.2 核心难点（面试深挖）
- **上下文共享**：子 Agent 需要多少上下文？太多烧 token、太少丢信息。
- **信任与治理**：子 Agent 权限最小化、审计、身份（`Agentic Identity & Trust`）。
- **失败恢复**：子 Agent 挂了主管怎么接管？(`supervisor` + 重试/回退)。
- **一致性**：多个 Agent 对同一实体/数据的认知一致（`Identity Graph`）。

本项目 `agency-agents/` 即一套**多 Agent / 专家 Agent** 目录：18 个 division（engineering/design/security/finance…），每个文件是一个带人格、流程、交付物、成功指标的专家 Agent；并支持安装到 Claude Code / Cursor / Codex / Gemini CLI / Qwen 等工具。

---

## 8. 状态机与持久化：LangGraph

LangGraph 解决了 LCEL 的短板：**跨轮次、有状态、含循环、可中断** 的复杂流。

### 8.1 核心概念
| 概念 | 说明 |
|------|------|
| `StateGraph` / `StateGraph(AgentState)` | 定义状态 schema |
| `MessagesState` | 预置的"消息列表"状态，自动 `add_messages` |
| 节点 `add_node` | 每个处理函数，`def node(state) -> dict` |
| 边 `add_edge` | 节点间流转；`add_conditional_edges` 条件路由 |
| `ToolNode` | 执行工具的内置节点 |
| `tools_condition` | "有 tool_calls 就去工具，否则结束"的判定 |
| `checkpointer` | 保存/恢复状态（记忆 + 中断） |
| `RetryPolicy` | 节点级重试策略 |
| `recursion_limit` | 递归上限，防死循环 |

### 8.2 本项目示例
```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([get_weather]))
builder.add_edge("agent", "tools", condition=tools_condition)
builder.set_entry_point("agent")
graph = builder.compile(checkpointer=checkpointer)   # 记忆
```

### 8.3 重试策略（`retry_demo.py`）
```python
from langgraph.types import RetryPolicy
retry = RetryPolicy(
    max_attempts=3,
    retry_on=should_retry,        # 自定义异常判断（如 429/502/503/504）
    initial_interval=1.0,
    max_interval=60.0,
)
builder.add_node("api_call", call_external_api, retry_policy=retry)
```

### 8.4 LCEL vs LangGraph（判断口诀）
> **"链管一个回合，图管一场战役。"**
> 单轮、无环、无共享可变状态 → LCEL；多轮/循环/Checkpoint/HITL/多 Agent → LangGraph。

---

## 9. 中间件与横切关注点

见 `LangChain/middleware/`（`AgentMiddleware`）。用于在 agent 主循环**不改主代码**地插拔能力。

### 9.1 全部 Hook
| Hook | 时机 | 用途 |
|------|------|------|
| `before_agent` / `abefore_agent` | 开始前（一次） | 初始化/鉴权 |
| `before_model` / `abefore_model` | 每次模型前 | 注入状态/日志 |
| `wrap_model_call` / `awrap_model_call` | 包裹模型调用 | 重试/回退/缓存/改写请求响应 |
| `after_model` / `aafter_model` | 每次模型后 | 记录响应 |
| `wrap_tool_call` / `awrap_tool_call` | 包裹工具调用 | 工具重试/校验/改写参数 |
| `after_agent` / `aafter_agent` | 结束时（一次） | 汇总/上报 |
| `dynamic_prompt` | — | 动态生成系统提示词 |

### 9.2 内置中间件（开箱即用）
`ModelRetryMiddleware` / `ModelFallbackMiddleware` / `ModelCallLimitMiddleware` /
`ToolRetryMiddleware` / `ToolCallLimitMiddleware` / `ToolErrorMiddleware` /
`ToolSelectionMiddleware` / `SummarizationMiddleware` / `TodoListMiddleware` /
`HumanInTheLoopMiddleware` / `PIIMiddleware` / `ContextEditingMiddleware` /
`FilesystemFileSearchMiddleware` / `ProviderToolSearchMiddleware` / `ShellToolMiddleware`。

---

## 10. 执行环境 / 沙箱（DeepAgents）

当 Agent 要"跑代码/执行命令"时，必须用**沙箱**做隔离，保证安全。

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

sandbox_backend = LocalShellBackend(root_dir="/")
agent = create_deep_agent(model=llm, backend=sandbox_backend,
                          system_prompt="你可以在沙箱中执行代码和命令。")
result = agent.invoke({"messages": [HumanMessage(content="运行 `ls -la`")]})
```

关键点：
- **执行策略**：`HostExecutionPolicy` / `DockerExecutionPolicy` / `CodexSandboxExecutionPolicy`（隔离强度不同）。
- **沙箱生命周期**：同一 `thread_id` 复用一个沙箱（`thread_scope.py`：`create_sandbox` / `connect_sandbox`，生产用 Redis/DB 存映射）。
- **为何要沙箱**：防止 Agent 执行任意代码破坏宿主、泄露密钥；做资源限制、网络隔离。

---

## 11. 模型工厂 / 多模型路由（ModelFactory）

`ai_app/model_factory/model_factory.py` 统一管理模型，支持本地/云多模型。

```python
ModelFactory().create_model()                       # 默认本地 agent 模型
ModelFactory().create_model(model_type='local', local_model_type='embed')  # 向量模型
ModelFactory().create_model(model_type='qwen')      # 云端 Qwen
```

| 类型 | 目标模型 | 端口 |
|------|---------|------|
| `agent` | Qwopus3.5-4B-Coder-MTP | 55002 |
| `coder` | qwen2.5-coder-7b | 55001 |
| `vl` | qwen2.5-vl-7b | 55003 |
| `embed` | nomic-embed-text | 55004 |

> 价值：把「模型选择/切换/成本/端到端地址」收敛到一处，**换模型零侵入**。生产里常配合**模型路由/回退**（主模型失败切备选）。

---

## 12. MCP 与工具生态

**MCP（Model Context Protocol）** 用统一方式接入外部工具/数据源，让工具"即插即用"、自由组合。

- **Server**：暴露工具（本地 `stdio` 或远程 `http`）。`demo02` 用 `FastMCP`。
- **Client**：连接多个 Server 取工具。`demo01`/`demo03` 用 `MultiServerMCPClient`。
- 本项目 `demo03` 经 WSL + `uvx es-mcp-server` 连 Elasticsearch，Agent 用自然语言查索引/文档。

```python
client = MultiServerMCPClient({
    "elasticsearch": {
        "transport": "stdio",
        "command": "wsl",
        "args": ["-d", "Ubuntu", "bash", "-lc", "export ES_HOST=localhost && ... uvx es-mcp-server"],
    }
})
tools = await client.get_tools()
agent = create_agent(llm, tools)   # 第4步：自然的自然语言查询
```

> 面试点：**MCP 是"工具分发层"**，让 Agent 免去每接一个数据源都要手写对接；`fastmcp` 可快速自定义 Server；`langchain-mcp-adapters` 充当 MCP↔LangChain 桥梁。

---

## 13. Agent 常见架构拓扑与选型

| 架构 | 复杂度 | 适用 | 典型库/模式 |
|------|--------|------|------------|
| 单 Agent + 工具 | 低 | 多数问答/工具场景 | `create_agent` |
| RAG Agent | 中 | 文档问答、知识库 | `rag_agents/` |
| 多 Agent（Supervisor/Router/Hierarchical） | 中高 | 复杂任务分解 | LangGraph / `agency-agents` |
| Agent + 沙箱 | 高 | 跑代码/命令/自动化 | DeepAgents |
| Agent + MCP 工具网络 | 中高 | 跨系统数据/工具接入 | langchain-mcp-adapters |
| Agent + HITL | 中高 | 需人工审核的关键业务 | `HumanInTheLoopMiddleware` |
| Agent + 记忆持久化 | 中高 | 长期记忆/多轮复杂流程 | LangGraph checkpointer |

**选型判据**（面试可答）：
1. 需不需要**循环/重试/自我修正**？要 → LangGraph / 中间件。
2. 需不需要**长记忆/跨轮**？要 → checkpointer + DB。
3. 需不需要**执行代码/系统操作**？要 → 沙箱。
4. 需不需要**多个专家分工**？要 → 多 Agent。
5. 需不需要**接很多外部系统**？要 → MCP。

---

## 14. 高频面试题速答

**Q1：Agent 和纯 LLM 的区别？**
> 纯 LLM 只输出文本；Agent 会输出 `tool_calls` 并进入「调用工具 → 回填 → 再调用 LLM」的循环，直到给出最终答案，因而能"真正行动"。

**Q2：ReAct 是什么？**
> 「边推理边行动」范式：`Thought → Action → Observation`，循环往复直到得出答案。LangChain `create_agent` 默认内核。

**Q3：Function Calling / Tool Calling 怎么实现的？**
> 把工具用 JSON Schema 描述（`@tool` 自动从签名/docstring 生成），模型按 schema 输出结构化参数；框架执行工具并把结果回填给模型。

**Q4：如何减少幻觉？**
> RAG（检索增强 + 强制"资料不足就承认不知道"）、精排（Reranker）、结构化输出、明确的 system_prompt 约束、提示注入防御。

**Q5：RRF 和 Reranker 的区别？**
> RRF 融合多路召回、只看排名、提召回率；Reranker 理解相关性、精排、提质、但慢且贵。

**Q6：多 Agent 怎么协作？有哪些拓扑？**
> Supervisor 主管、Router 路由、Hierarchical 层级、Sequential 串行、Parallel 并行、Network 竞争辩论、Swarm 群体。难点在上下文/信任/故障恢复/一致性。

**Q7：Agent 死循环怎么办？**
> `recursion_limit`、步数上限、`ToolCallLimitMiddleware`/`ModelCallLimitMiddleware`、`todo` 清单、终止条件（`jump_to="end"`）。

**Q8：Agent 调外部系统不安全，如何防护？**
> 沙箱（Sandbox）、最小权限工具、`wrap_tool_call` 校验、HITL 人工确认、PII 脱敏、审计日志、提示注入防线。

**Q9：Agent 如何做可观测与成本控制？**
> 回调收集 `usage_metadata`（token）、归因到 `run_id`/`tag`；限流 `max_concurrency`；缓存语义命中；熔断 + 降级话术；单请求 + 月度预算护栏。

**Q10：什么时候用 LCEL，什么时候用 LangGraph？**
> LCEL 管单轮线性流；LangGraph 管多轮/循环/checkpoint/HITL/多 Agent。

**Q11：如何评测一个 Agent？**
> 端到端任务成功率、工具调用准确率、召回率/NDCG（RAG）、答案正确性/事实一致性、防幻觉、成本/时延；用离线基准 + 在线指标。

---

## 15. 生产级考量：可观测 / 安全 / 成本 / 评测

### 15.1 可观测
- **Callbacks / Tracing**：LangSmith / LangFuse / 自研；一次请求一棵 trace 树，按 `run_id`、`tag`、`metadata` 聚合。
- **指标**：`ttfb`（首 token 时延）、总时延、`input/output_tokens`、fallback 触发率、缓存命中率、错误率、P95。
- **日志**：结构化日志带 `run_id`，与 trace 系统双勾稽。

### 15.2 安全
- **提示注入防线**：不可信文本加定界符 + 声明忽略其中指令；工具最小权限；危险操作二次确认（HITL）。
- **密钥**：仅经环境变量/Secret Manager 注入，绝不硬编码（本项目 `.env`、`fail-fast` 校验）。
- **PII 合规**：上模前 `RunnableLambda` DLP 脱敏（可接 Presidio）。
- **沙箱**：代码/命令执行走沙箱。
- **审计**：Agent 身份、权限、操作留痕（`Agentic Identity & Trust`）。

### 15.3 成本与降级
```
L0 缓存命中（精确/语义）      → 毫秒返回、成本最优
L1 轻量模型顶替重型模型        → 质量微降、可用性保全
L2 跨供应商 fallback           → 抗单云故障
L3 熔断 + 静态兜底话术/转人工   → 保品牌底线
```
每级独立监控告警；降级率突增 = 事故信号。

### 15.4 评测（Evaluation）
- **离线**：基准集 + 指标（RAG：召回率/MRR/NDCG；对话：事实一致性/有用性）。
- **在线**：任务成功率、用户反馈、成本、时延、兜底触发率。
- **对抗**：幻觉注入、提示注入、边界用例。

---

## 16. 一页带走

- **Agent = LLM + Tools + Memory + Loop**。
- **四大范式**：ReAct / Plan-Execute / Reflection / Multi-Agent。
- **两条主线**：一条是**能力**（工具、RAG、多 Agent、MCP、沙箱）；一条是**工程**（状态机、记忆、中间件、可观测、安全、成本）。
- **核心口诀**：
  - *链管一个回合，图管一场战役。*
  - *RRF 召回提率，Reranker 精排提质。*
  - *先想清楚要不要循环，再决定上不上图。*
  - *Agent 能力越强，越要把安全/成本/评测当一等公民。*

---

> 相关阅读（本项目已产出）：
> - `LangChain/middleware/middleware总结.md` —— Agent 中间件全部 hooks
> - `LangChain/MCP/MCP总结.md` —— MCP 协议与接入
> - `main/rag_agents/rrf_rerank.md` —— RRF 与 BGE Reranker
> - `prompt.md` —— LCEL 与 Runnable 协议手册

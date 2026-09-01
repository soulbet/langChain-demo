# 🔬 LangChain / LangGraph / DeepAgents 面试级总结

> 版本基线（本机环境实测）：`langchain 1.3.14`、`langchain-core 1.4.9`、`langgraph`（当前）、`deepagents 0.6.x`
> 说明：本文基于**框架领域知识点**整理，不依赖任何项目内代码。
> 目标：把三大框架做成可背、可讲、可答、可场景化迁移的面试手册。

---

## 目录

- [Part 0｜三者关系总览](#part-0三者关系总览)
- [Part 1｜LangChain](#part-1langchain)
- [Part 2｜LangGraph](#part-2langgraph)
- [Part 3｜DeepAgents](#part-3deepagents)
- [Part 4｜三者集成与选型](#part-4三者集成与选型)
- [Part 5｜关键词大表](#part-5关键词大表)
- [Part 6｜生产级考量](#part-6生产级考量)
- [Part 7｜高频面试题速答](#part-7高频面试题速答)

---

## Part 0｜三者关系总览

```
┌───────────────────────────────────────────────────────────┐
│  应用层：DeepAgents（高层次多智能体/沙箱脚手架）                │
│           create_deep_agent / SubAgent / Filesystem / 记忆  │
├───────────────────────────────────────────────────────────┤
│  编排层：LangGraph（有状态、含循环、可持久化的图引擎）            │
│           StateGraph / Node / Edge / checkpointer / 流式     │
├───────────────────────────────────────────────────────────┤
│  表达层：LangChain / LangChain-Core（LCEL + Runnable 协议）   │
│           prompt | llm | parser / tools / retriever 组合     │
├───────────────────────────────────────────────────────────┤
│  基座层：ChatModels / Embeddings / VectorStores / Document    │
│           / OutputParsers / Callbacks / Tracking(Tracing)    │
└───────────────────────────────────────────────────────────┘
```

**一句话区分**：
- **LangChain**：提供"组件 + 组合方式"，核心是 **LCEL（`|` 管道的 Runnable 协议）**。
- **LangGraph**：提供"有状态 + 含循环 + 可持久化"的**图引擎**，管理 Agent 的**流程与状态**（比 LCEL 更强）。
- **DeepAgents**：基于 LangGraph / LangChain 之上的一层**高层次脚手架**，开箱即用"**能跑代码/操作文件/拆子任务/有记忆的深度 Agent**"。

> 记忆口诀：**LangChain 管"组件怎么拼"，LangGraph 管"流程怎么走"，DeepAgents 管"深度 Agent 怎么开箱即用"。**

---

## Part 1｜LangChain

### 1.1 定位
LangChain 是构建 LLM 应用的**编排框架**，把模型、提示、解析器、检索器、工具等封装成**统一协议（Runnable）**，用声明式方式组合成链/Agent。

### 1.2 生态包拆分（面试常考）
| 包 | 职责 |
|----|------|
| `langchain-core` | 最底层抽象：`Runnable`、`BaseChatModel`、`BaseTool`、`Document`、`BaseRetriever`、`BaseCallbackHandler`、`RunnableConfig` |
| `langchain`（主包） | v1.0 起只保留 prompts/models/tools/agents；`create_agent` 底层由 LangGraph 驱动 |
| `langchain-community` | 第三方集成（vector store、document loader、text splitter 等） |
| `langchain-openai` / `-anthropic` / `-google-genai` / `-tavily` | 各家模型/工具适配（实现 `BaseChatModel`） |
| `langchain-mcp-adapters` | MCP ↔ LangChain 适配 |
| `langchain-text-splitters` | 文本切分器 |

### 1.3 LCEL 与 Runnable 协议（核心考点）

**LCEL（LangChain Expression Language）** 是用管道 `|` 组合组件的声明式 DSL，让所有组件实现同一 `Runnable` 协议。

**Runnable 统一接口**：`invoke / ainvoke / batch / abatch / stream / astream / astream_events`。
- 自定义组件**只需实现 `invoke`**，其余方法有默认实现兜底：
  - `batch` 默认用线程池并发跑 N 个 `invoke`；
  - `stream` 默认退化为一次性 `invoke` 后整块返回（组件可覆写为真流式）；
  - 异步方法默认通过线程池**自动桥接**到同步版本。

**组合原语**：
| 原语 | 作用 | 关键点 |
|------|------|--------|  
| `\|` / `RunnableSequence` | 串行：前一步输出 = 后一步输入 | `__or__/__ror__` 自动 `coerce`（`str→PromptTemplate`、`Callable→RunnableLambda`、`dict→RunnableParallel`） |  
| `RunnableParallel` / `{...}` | 并行分支，同输入喂给所有分支合并为 dict | 短路：任一失败整体失败（除非分支自带降级） |  
| `RunnablePassthrough` | 恒等透传 | 几乎总配合 `.assign()` |  
| `.assign(**fns)` | 增量追加键 | 按定义顺序依次求值，后面的能读到前面刚算的键 |  
| `RunnableLambda` | 包装一个普通函数成组件 | 返回生成器可参与流式；可传 `afunc=` 提供原生异步 |  
| `RunnableBranch` | 顺序条件路由 | 条件按顺序短路；最后一个为默认分支 |  
| `.with_retry()` | 指数退避重试 | 底层 tenacity；只重试 `retry_if_exception_type` |  
| `.with_fallbacks()` | 失败切备选 | 触发即切，**不做退避等待**（重试 vs 降级的区别） |  
| `.bind()` | 固定传给末端的 kwarg | 经典：`llm.bind(tools=[...])`、`stop=` |  
| `.with_config()` / `.with_listeners()` | 打 tag/run_name；挂生命周期回调 | 便于 trace 归因 |  
| `.pick()` / `itemgetter` | 抽取 dict 字段 | 把上游 dict 拆给不同分支 |  
| `.configurable_fields()` | 运行时可切参数 | 配合 `config={"configurable": {...}}` 多租户热切换 |  
| `RunnableWithMessageHistory` | 自动注入/持久化会话历史 | `get_session_history` 回调 |  

**RunnableConfig 关键字段**：`callbacks`、`tags`、`metadata`、`run_name`、`max_concurrency`、`recursion_limit`（默认 25）、`configurable`、`run_id`。

**前端深挖点（面试常见）**：
1. **`|` 背后做了什么**：`Runnable.__or__` 内部 `coerce()` 把右侧强制转换后包装成 `RunnableSequence(first, last)`。
2. **`stream` 与 `invoke` 差异**：`stream` 先流式拉首步输出，再尝试让后续组件以 `transform(chunk_iter)` 增量消费；组件不支持增量才回退为"攒齐再 invoke"——这就是加不加 `StrOutputParser` 都不影响打字机效果的原因。
3. **同步函数在 `ainvoke` 里不会卡事件循环**（被丢线程池），但会占 worker 线程；高性能用 `RunnableLambda(fn, afunc=afn)`。

### 1.4 核心组件

#### Chat Models
- 抽象 `BaseChatModel`，每个供应商实现它。
- 绑定工具：`model.bind_tools([...])`；结构化输出：`model.with_structured_output(PydanticSchema)`（强约束为工具 schema）。
- 关键参数：`model`、`temperature`、`max_tokens`、`timeout`、`max_retries`、`callbacks`。

#### Prompts
- `PromptTemplate` / `ChatPromptTemplate.from_messages([("system",...),("human",...)])`；
- `MessagePlaceholder`（占位插入历史/检索内容）；
- Few-shot（`FewShotChatMessagePromptTemplate` + `ExampleSelector`）。

#### Output Parsers
- `StrOutputParser` / `PydanticOutputParser` / `JsonOutputParser`；
- 现代更推荐 `with_structured_output(schema)` / structured output（工具强约束），或 `astream_events` 解析。

#### 文档与向量
- Document Loaders（PDF/网页/数据库/文件…）→ `Document(page_content, metadata)`；
- Text Splitters：`RecursiveCharacterTextSplitter`（`chunk_size` / `chunk_overlap` / `separators`）、字符/语义切分；
- Embeddings：`embed_documents` / `embed_query`，文本→向量；
- Vector Stores：PGVector / FAISS / Chroma / Qdrant 等（`similarity_search`、`as_retriever`、`add_texts`）；
- Retriever 增强：MultiQuery / Ensemble / ContextualCompression / SelfQuery（查询改造、多路召回、压缩）。

#### Tools
- `@tool`（从签名+docstring 自动生成 schema）；`StructuredTool`；
- `ToolNode`（LangGraph 里执行工具）；`tool_choice` 强制/选择；
- 工具错误处理：`handle_tool_errors`、`ToolErrorMiddleware`。

#### Callbacks / Tracing
- `BaseCallbackHandler`：`on_llm_start/on_llm_end/on_chain_start/on_tool_start...`；
- 读标准化的 `usage_metadata` 统计 token；接 LangSmith / LangFuse 做观测。

### 1.5 Agent（v1 `create_agent`）
```python
from langchain.agents import create_agent
agent = create_agent(model, tools=[...], system_prompt="...", middleware=[...])
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```
- 底层由 **LangGraph** 驱动（`create_react_agent` 内核）。
- `middleware` 支持挂载横切逻辑（见 Part 2 中间件一节）。
- 消息流：模型输出 `tool_calls` → 执行工具 → 回填 → 再次模型 → … → 最终答案。

### 1.6 Memory 与 RAG
- **Memory**：`RunnableWithMessageHistory`（进程内）；跨线程持久用 LangGraph `checkpointer`。
- **RAG**：`Embedding → Retriever(Vector/BM25/混合) → 上下文拼装 → LLM`。
  - 混合检索：向量 + 关键词（BM25），用 **RRF** 融合；
  - 精排：**Reranker**（CrossEncoder）提高准确率；
  - 核心痛点：**幻觉**（用"检索增强 + 资料不足就承认不知道"缓解）。

### 1.7 版本演进（高频）
| 版本 | 时间 | 关键变化 |
|------|------|---------|
| v0.0.x | 2023.08 | LCEL 实验特性；`Chain` 类为主流 |
| v0.1 | 2024.01 | 拆分出 `langchain-core`；`Runnable` 定型 |
| v0.2 | 2024.05 | `langchain-community` 解耦；`astream_events` v2 |
| v0.3 | 2024.09 | Pydantic v2 全面采用 |
| v1.0 | 2025.10 | 主包瘦身；遗留 `Chain`（`LLMChain` 等）移除，迁移到 `langchain-classic`；`create_agent` 底层改为 LangGraph；消息用 `content_blocks` |

**迁移对照（面试常考）**：
| 旧 API | 现代替代 |
|--------|----------|
| `LLMChain` | `prompt \| llm \| StrOutputParser()` |
| `SequentialChain` | `chain_a \| chain_b` |
| `ConversationChain` | `RunnableWithMessageHistory` 或 LangGraph checkpointer |
| `RetrievalQA` | `create_retrieval_chain(...)` |
| `ConversationalRetrievalChain` | `history_aware_retriever` + `create_retrieval_chain` |
| `RouterChain` | `RunnableBranch` 或语义路由 |
| `AgentExecutor` | LangGraph 预置 `create_agent` |

### 1.8 常见场景
- 对话/客服、RAG 知识库问答、结构化抽取、动态路由（不同意图→不同专家链）、多模型回退/重试、带记忆的会话。

---

## Part 2｜LangGraph

### 2.1 定位
LangGraph 是专门给 Agent 场景设计的**低层有状态图引擎**，解决了 LCEL 的三大短板：**多轮、循环、共享可变状态**。
用**图（Graph）**的方式把 Agent 流程建模为：**节点(Node) + 边(Edge) + 状态(State) + 检查点(Checkpointer)**。

### 2.2 核心概念
| 概念 | 说明 |
|------|------|
| `StateGraph` | 图的构造器，接收一个 State 类型 |
| `State` | 图状态：一个 `TypedDict`，字段用 `Annotated[type, reducer]` 定义"如何合并" |
| `Node` | 一个接收 `state`、返回 `dict`（部分状态合并）的函数 |
| `Edge` | 节点间流转；`add_conditional_edges` 条件路由 |
| `Command` | 命令对象：跨节点跳转（`goto` / `update` / `graph`），用于多 Agent 控制权移交 |
| `Reducer` | 状态合并函数（如 `add_messages`） |
| `Checkpointer` | 保存/恢复图状态（记忆 + 断点 + 时间旅行） |
| `Thread` | 用 `thread_id` 标识的一场对话/会话 |
| `interrupt` | 用于 HITL（人工介入后 `Command(resume=...)` 恢复） |
| `RetryPolicy` | 节点级重试 |
| `recursion_limit` | 递归上限（默认 25），防死循环 |

### 2.3 状态管理
```python
from langgraph.graph import StateGraph, MessagesState
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(MessagesState):      # MessagesState 预置 messages + add_messages
    current_city: str                 # 自定义业务状态
    # 可加 reducer： Annotated[list, operator.add] 等
```
- **Reducer 决定合并语义**：`add_messages` 是追加；`operator.add` 是列表累加；自定义 reducer 可做覆盖/去重。
- 状态更新由节点返回的 dict **部分合并**（不是整体替换）。

### 2.4 控制流
```python
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([tool]))
builder.add_edge("agent", "tools", condition=tools_condition)  # 有 tool_calls 才去工具
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", router_fn, {"continue": "agent", "end": END})
graph = builder.compile(checkpointer=checkpointer)
```
- **条件边**根据返回的 key 决定下一节点；
- **`Command(goto=...)`** 可在节点内直接"跳转"，实现多 Agent 之间的控制权移交；
- **循环**：节点回连自身（反思、重试、多步）。

### 2.5 持久化与记忆
```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()                 # 测试用；生产换 Sqlite/Postgres/Redis
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [("user","hi")]}, config=config)
```
- `thread_id` 相同的多次调用**自动从检查点恢复**上下文；
- `get_state` / `update_state` 查看/改写状态；**Time travel**（回到历史 step）用于调试/重放；
- 生产建议 **PostgresSaver / RedisSaver**（`checkpoint.memory` 只是进程内）。

### 2.6 工具调用
- `ToolNode`：执行工具；`tools_condition`：判断是否有 `tool_calls`；
- `create_react_agent(model, tools, ...)`：快速构建 ReAct 型 Agent（v1 `create_agent` 底层即此）；
- `InjectedState` / `InjectedStore`：向工具注入状态/存储；`ToolCallTransformer`/`ValidationNode`：工具转换与校验。

### 2.7 重试与容错
```python
from langgraph.types import RetryPolicy
builder.add_node("api_call", fn, retry_policy=RetryPolicy(
    max_attempts=3, retry_on=lambda e: isinstance(e, (TimeoutError,)), 
    initial_interval=1.0, max_interval=60.0))
```
也可用 `.with_retry()` / `.with_fallbacks()`；配合 `Command` 做"某节点失败→走兜底分支"。

### 2.8 流式
```python
for chunk in graph.stream(input, config, stream_mode="values"):   # 每步完整状态
for chunk in graph.stream(input, config, stream_mode="updates"):  # 每步的增量更新
for chunk in graph.stream(input, config, stream_mode="messages"): # token 级（配合中间件 transformers）
for chunk in graph.stream(input, config, stream_mode="custom"):   # 自定义事件（runtime.stream_writer）
```
- `astream` / `astream_events` 提供细粒度事件流，适合打 UI 打字机 + 过程可视化。

### 2.9 多 Agent
LangGraph 原生支持多 Agent 编排：
- **Supervisor / 主管**：一个 Agent 决定把任务分给哪个子 Agent；
- **Subgraph**：把子图作为节点，实现层级/分层；
- **`Command(goto=...)`**：控制权在 Agent 之间显式移交；
- **`send()`**：扇出并行执行多个子任务再汇总；
- 常见拓扑：Supervisor / Hierarchical / Router / Sequential / Parallel / Network(辩论) / Swarm。

### 2.10 常见场景
- 多步/循环流程、长任务、自我反思、人审(HITL)、长记忆会话、多 Agent 协作、可中断恢复的自动化流程。

---

## Part 3｜DeepAgents

### 3.1 定位
**Deep Agents（深度智能体）** 是一类"**能在沙箱中实际执行任务**"的 Agent：不仅能聊天、调用工具，还能读写文件、运行 shell 命令、拆解子任务、自我反思，并带走长记忆。
**DeepAgents = 基于 LangGraph/LangChain 的高层次脚手架**，让你用很少的代码得到一个"开箱即用"的深度 Agent。

> 一句话：**普通 Agent 帮你"回答"，Deep Agent 帮你"做事"。**

### 3.2 创建
```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=llm,
    tools=[my_tool],               # 额外自定义工具（可选）
    system_prompt="...",
    # 内置能力：Filesystem（文件系统）、Shell（命令）、Subagents（子任务）、Memory（记忆）
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

### 3.3 核心内置能力（默认开启）
| 能力 | 说明 | 相关组件 |
|------|------|---------|
| **文件系统** | 读写/搜索/列出项目文件 | `FilesystemMiddleware`（+ `FilesystemPermission` 权限控制） |
| **执行命令** | 在沙箱里运行 shell/代码 | `LocalShellBackend`（沙箱后端） |
| **子智能体** | 拆解复杂任务交给子 Agent | `SubAgent` / `SubAgentMiddleware` |
| **记忆** | 跨轮次/跨会话记忆 | `MemoryMiddleware` |
| **反思/自评** | 按 rubric 自评与改进 | `RubricMiddleware` |

### 3.4 沙箱与后端（backends）
`create_deep_agent(backend=...)` 决定"命令/文件在哪里执行、隔离多强"：
| 后端 | 说明 |
|------|------|
| `LocalShellBackend` | 本地 shell（最简单，隔离弱） |
| `FilesystemBackend` | 虚拟/受限文件系统 |
| `ContextHubBackend` | 云端受限执行环境 |
| `LangSmithSandbox` | 与 LangSmith 集成的沙箱 |
| `StateBackend` | 基于图状态的执行 |
| `StoreBackend` | 基于持久化 store 的执行 |
| `CompositeBackend` | 组合多种后端 |

还有顶层 `BackendProtocol`（自定义后端协议）、`BackendContext`、`DEFAULT_EXECUTE_TIMEOUT`（执行超时）。

> 为什么必须沙箱：**隔离 + 安全**，防止 Agent 执行任意代码破坏宿主、泄露密钥；可做资源限制、网络隔离、文件作用域。

### 3.5 沙箱生命周期与线程
- 每个沙箱有唯一 `sandbox_id`；同一 `thread_id` 可复用同一沙箱，保证"文件/状态持续存在"；
- 生产环境用 Redis/DB 存 `thread_id → sandbox_id` 映射；
- **线程（thread/thread_id）**：多轮对话的会话标识，配合 LangGraph checkpointer 记忆与沙箱复用。

### 3.6 多 Agent / 子智能体
- `SubAgent` / `AsyncSubAgent`：定义一个子智能体（独立任务、独立上下文）；`SubAgentMiddleware` 把它接入主 Agent；
- `DeepAgentState`：深度 Agent 的状态 schema（含 subagents 等）。
- 适用：把"大任务"拆成"可并行/可分派的子任务"，主 Agent 负责调度与汇总。

### 3.7 Profiles（画像/配置）
- `ProviderProfile` / `HarnessProfile` / `HarnessProfileConfig`（适配不同宿主/工具链）；
- `register_provider_profile` / `register_harness_profile`：注册自定义画像，让同一 Agent 匹配多套运行环境（如不同供应商/宿主）。

### 3.8 常见注意点
- **成本**：深度 Agent 会多次调用模型 + 跑子任务 + 文件/命令，token 与时延显著高于普通 Agent，需设 **执行超时 / 调用上限 / 预算**。
- **安全**：沙箱要最小权限；命令执行要限制作用域；敏感操作建议 HITL。
- **可观测**：深度 Agent 步骤多，务必接 tracing 并按 `run_id`/`tag` 归因，便于排障与成本分析。

---

## Part 4｜三者集成与选型

### 4.1 集成方式
```
LangChain(组件)  ──组合(Runnable)──▶  LangGraph(流程/状态)  ──脚手架──▶  DeepAgents(深度Agent)
```
- **LangChain** 组件天然是实现 LangGraph 节点/工具的材料；
- **LangGraph** 是 `create_agent` / `create_deep_agent` 的底层引擎；
- **DeepAgents** 复用上述两者，只暴露高层配置（`backend`/`tools`/`system_prompt`）。

### 4.2 选型判据
| 需求 | 选哪个 |
|------|--------|
| 单轮线性链、轻量 | LangChain LCEL |
| 多轮/循环/反思/多 Agent/持久化/HITL | LangGraph |
| 要"能跑代码/操作文件/拆子任务/带记忆/自评" | DeepAgents |
| 要接很多外部系统/工具 | LangChain Tools + MCP |

> 口诀：**链管一个回合，图管一场战役；深智能体把"做事"也交给 AI。**

---

## Part 5｜关键词大表

| 关键词 | 含义 |
|--------|------|
| LCEL | LangChain Expression Language，`\|` 管道组合的声明式 DSL |
| Runnable | 统一协议（invoke/stream/batch/async），组件可互换 |
| RunnableSequence / Parallel / Passthrough / Branch | 串行 / 并行 / 透传 / 条件路由 组合原语 |
| RunnableConfig | 运行配置（callbacks/tags/metadata/recursion_limit/max_concurrency） |
| with_retry / with_fallbacks | 重试（退避）/ 降级（切换备选） |
| chat model | 语言模型接口（BaseChatModel） |
| bind_tools / with_structured_output | 绑定工具 / 结构化输出 |
| PromptTemplate / ChatPromptTemplate | 提示模板 |
| StrOutputParser / PydanticOutputParser | 输出解析 |
| Document / TextSplitter | 文档对象 / 文本切分（chunk_size/overlap/separators） |
| Embeddings / VectorStore / Retriever | 向量化 / 向量库 / 检索器 |
| RAG / Hybrid Search / RRF / Reranker | 检索增强 / 混合检索 / 排名融合 / 精排 |
| Tool / ToolNode / tools_condition / tool_choice | 工具定义 / 工具执行节点 / 条件判定 / 强制选择 |
| Callbacks / Tracing / LangSmith | 回调 / 追踪 / 观测 |
| StateGraph / State / Node / Edge / Command | 图 / 状态 / 节点 / 边 / 命令 |
| Reducer / Annotated | 状态合并器 / 类型注解（定义 reducer） |
| MessagesState / add_messages | 预置消息状态 / 追加 reducer |
| Checkpointer / thread_id / Time travel / interrupt | 检查点 / 线程 / 时间旅行 / 中断（HITL） |
| create_react_agent / create_agent | ReAct 型 Agent 快速构建 |
| RetryPolicy / recursion_limit | 节点重试策略 / 递归上限 |
| stream_mode (values/updates/messages/custom) | 流式模式 |
| Supervisor / Hierarchical / send / Subgraph | 多 Agent 拓扑与扇出 |
| create_deep_agent | 深度 Agent 创建入口 |
| SubAgent / SubAgentMiddleware / DeepAgentState | 子智能体 / 接入中间件 / 状态 |
| FilesystemMiddleware / FilesystemPermission | 文件系统能力 / 权限控制 |
| MemoryMiddleware / RubricMiddleware | 记忆 / 按标准自评 |
| LocalShellBackend / FilesystemBackend / CompositeBackend | 沙箱后端（本地/受限/组合） |
| BackendProtocol / BackendContext / sandbox_id | 后端协议 / 上下文 / 沙箱 id |
| ProviderProfile / HarnessProfile | 供应商画像 / 宿主画像 |
| MCP / MultiServerMCPClient / FastMCP | 模型上下文协议 / 多端客户端 / 快速定义服务 |
| HITL / human_in_the_loop | 人工介入 |
| Hallucination / Prompt Injection | 幻觉 / 提示注入 |

---

## Part 6｜生产级考量

### 6.1 可观测
- Tracing（LangSmith/LangFuse）、`run_id` + `tag` + `metadata` 归因；
- 指标：`ttfb`、总时延、`input/output_tokens`、fallback/重试触发率、缓存命中率、错误率。

### 6.2 安全
- 提示注入防线（不可信文本定界 + 忽略其中指令）；
- 工具最小权限；高危操作 HITL / 沙箱；
- 密钥只走环境变量/Secret；PII 上模前脱敏；
- Deep Agent 一定要**沙箱 + 执行超时 + 资源限制**。

### 6.3 成本与降级
- 层级：L0 缓存 → L1 小模型替大模型 → L2 跨供应商 fallback → L3 熔断+兜底话术/转人工；
- 护栏：单请求 token 上限 + 月度预算熔断；`max_concurrency` 抗限流；
- DeepAgent 成本高：限制调用步数/子任务数，考虑摘要压缩历史。

### 6.4 评测
- 离线：任务成功率、工具调用准确率、RAG 召回/NDCG、正确性/一致性、防幻觉；
- 在线：成功率、用户反馈、成本、时延、兜底率；对抗：幻觉/提示注入/边界用例。

---

## Part 7｜高频面试题速答

**Q1：LCEL 为什么能淘汰 LLMChain？**
> LLMChain 把组合固化在类层次里，需求一变就要继承新类；LCEL 把组合下放为数据（DAG 对象），7 种原语即可覆盖旧几十个 Chain 的功能，且流式/异步/重试在协议层一次性解决——"类爆炸 → 协议收敛"。

**Q2：`\|` 运算符背后发生了什么？**
> `Runnable.__or__/__ror__` 内部 `coerce()`：字典→`RunnableParallel`、函数→`RunnableLambda`、字符串→`PromptTemplate`，再包装为 `RunnableSequence(first, last)`。所以 `"tpl" \| fn \| llm` 混合写法合法。

**Q3：`stream` 和 `invoke` 的区别？**
> `invoke` 是 for 循环步步传递；`stream` 先流式拉首步，再试着让后续组件以 `transform(chunk_iter)` 增量消费，遇到不支持的才回退为"攒齐再 invoke"。也是 `StrOutputParser` 不打断打字机的原因。

**Q4：RunnableParallel 与 `.assign()` 的区别？**
> Parallel 把同输入复制给所有分支、分支互相不可见、覆盖式合并；`assign` 在一个 dict 上顺序追加键，后来的函数能读到前面刚算出的键，适合"逐步增强上下文"。

**Q5：`with_retry` 和 `with_fallbacks` 的触发顺序？写成 `(primary.with_retry()).with_fallbacks([backup.with_retry()])`？**
> 内层先做完所有重试仍失败，才向外触发 fallback；backup 的重试独立执行各自的 N 次。总最坏时长 ≈ 主重试串行和 + 备重试串行和，须据此设整体超时。

**Q6：什么时候该用 LangGraph 而不是 LCEL？**
> 出现 (a) 循环/自我修正迭代；(b) 跨请求持久状态；(c) 人审中断-恢复；(d) 多 Agent 控制权移交 任一成立即上 LangGraph。

**Q7：LangGraph 的状态怎么合并？**
> State 字段用 `Annotated[type, reducer]` 定义合并语义（`add_messages` 追加、`operator.add` 累加、自定义覆盖）；节点返回 dict 做**部分合并**而非整体替换。

**Q8：`ToolNode` 和 `tools_condition` 怎么配合？**
> `tools_condition` 判断模型输出是否含 `tool_calls`：有则路由到 `ToolNode` 执行工具，否则走下一步/结束；这就是 ReAct 循环的核心。

**Q9：LangGraph 怎么做持久记忆 / HITL？**
> 编译传 `checkpointer`，用 `thread_id` 区分会话；`interrupt` 用于暂停等人审，之后 `Command(resume=...)` 恢复。生产用 Postgres/Redis 而非内存。

**Q10：Deep Agent 和普通 Agent 的区别？**
> 普通 Agent 会调工具/检索；Deep Agent 还会**在沙箱里跑命令、读写文件、拆子任务、带记忆、按 rubric 自评**，即"能真正做事"。代价是成本/时延更高，要设沙箱 + 超时 + 预算。

**Q11：为什么要沙箱？有哪些后端？**
> 隔离 + 安全（防任意代码破坏、密钥泄露），可做资源/网络/文件作用域限制。后端：`LocalShellBackend`/`FilesystemBackend`/`ContextHubBackend`/`LangSmithSandbox`/`StateBackend`/`StoreBackend`/`CompositeBackend`。

**Q12：子智能体（SubAgent）解决什么？**
> 把大任务拆成可并行/可分派的子任务，主 Agent 调度汇总；降低单 Agent 上下文压力、提升专业化与可维护性（`SubAgentMiddleware`/`DeepAgentState`）。

---

## 一页带走

- **LangChain**：`组件 + LCEL（Runnable 协议）`，用 `\|` 拼装。
- **LangGraph**：`有状态 + 循环 + 可持久化 + 多 Agent` 的图引擎。
- **DeepAgents**：`沙箱 + 文件 + 子任务 + 记忆 + 自评` 的深度 Agent 脚手架。
- **核心口诀**：
  - *链管一个回合，图管一场战役，深智能体把"做事"也交给 AI。*
  - *LCEL 强在组合，LangGraph 强在流程与状态，DeepAgents 强在"开箱即用地做事"。*
  - *工具给它手脚，记忆给它连续性，沙箱给它安全边界，图给它可控流程。*

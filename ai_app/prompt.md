收到！你的模板中 **`[知识点名称]` 占位符没有填写**。作为面试最高频、也是理解现代 LangChain 一切设计的基石，我默认为你输出 **「LCEL（LangChain Expression Language）与 Runnable 协议」** 的完整手册——掌握它，等于掌握了 LangChain 的骨架。

---

# 📘 LangChain 面试级知识点手册：LCEL 与 Runnable 协议

> 适用版本：`langchain-core >= 0.1` 至 `1.x`（差异处会显式标注）

---

## 📖 概述

**一句话定义**：LCEL 是构建在 `Runnable` 统一协议之上的声明式组合 DSL，用管道符 `|` 把 Prompt、模型、解析器、检索器等组件编排成一条可流式、可异步、可重试的执行链。

**设计目标与解决的核心问题**：

| 传统写法的痛点（旧 `Chain` 类） | LCEL 的解法 |
|---|---|
| 每种链（`LLMChain`/`RetrievalQA`…）都是独立黑盒，能力不互通 | 所有组件实现同一个 `Runnable` 协议，任意组件可互换拼装 |
| 流式、异步、批量需要每个类单独实现一遍 | 协议层统一提供：实现一次 `invoke`，自动获得 `stream`/`batch`/`ainvoke` 等 |
| 组合逻辑散落在子类的 `_call` 里，难以测试与复用 | 组合关系成为**数据**（可序列化的 DAG），而非**代码** |
| 无法优雅处理降级、重试、路由等横切关注点 | `.with_fallbacks()` / `.with_retry()` / `RunnableBranch` 一行接入 |

**在生态中的位置**：

- `langchain-core.runnables` 是全生态的最底层抽象，被 chat models、retrievers、tools、agents 全部依赖；
- 自 LangChain **v0.1（2024.01）** 起逐步取代遗留 `Chain` 类；**v1.0（2025.10）中遗留 Chain 已彻底移除**，LCEL 成为唯一的链式表达方式；
- 边界：LCEL 负责**单次请求内的无环数据流（DAG）**；跨轮次、有状态、含循环的工作流交给 **LangGraph**。

---

## 🏗️ 核心架构

### 架构层次（自底向上）

```text
┌─────────────────────────────────────────────────────────────┐
│ 应用层：create_agent (v1.x) / langserve / 用户业务链              │
├─────────────────────────────────────────────────────────────┤
│ 组合原语层：RunnableSequence │ RunnableParallel │ RunnableLambda │
│            RunnablePassthrough(.assign) │ RunnableBranch      │
│            RunnableWithFallbacks │ RunnableRetry (装饰器产物)   │
├─────────────────────────────────────────────────────────────┤
│ 统一协议层：Runnable 抽象基类                                    │
│   同步: invoke / batch / stream                               │
│   异步: ainvoke / abatch / astream / astream_events(v2)        │
├─────────────────────────────────────────────────────────────┤
│ 原语适配层：chat models / LLMs / prompts / retrievers /         │
│           output parsers / tools —— 全部实现 Runnable          │
├─────────────────────────────────────────────────────────────┤
│ 支撑设施：RunnableConfig(ctx) / Callbacks(观测) /               │
│          Tracing(LangSmith) / 序列化(dumpd, 供 LangServe)      │
└─────────────────────────────────────────────────────────────┘
```

### 关键抽象：`Runnable`

`langchain_core.runnables.base.Runnable` 是一个泛型抽象类 `Runnable[Input, Output]`，核心约定：

- **最小实现义务只有同步路径**：自定义组件只需实现 `invoke()`；
- 其余方法有默认实现兜底：
  - `batch()` 默认用线程池并发跑 N 个 `invoke`；
  - `stream()` 默认退化为“一次性 `invoke` 后整块返回”（组件可覆写为真流式）；
  - 全部异步方法默认通过 `run_in_executor` **自动桥接**到同步版本（反之亦然）。
- 关键方法族：
  - **运行**：`invoke / batch / stream / astream / astream_events`；
  - **装饰**：`.bind()` / `.with_config()` / `.with_retry()` / `.with_fallbacks()` / `.with_listeners()` / `.with_types()`；
  - **动态化**：`.configurable_fields()` / `.configurable_alternatives()`。

### 数据流转过程

`a | b | c` 在**构造期**就生成了一个 `RunnableSequence(a, b, c)` 对象，此后每次调用：

1. **invoke 路径**：输入依次穿过每一步，前一步输出直接作为后一步输入；
2. **stream 路径（面试深挖点）**：首步产出 chunk 流，序列尝试让后续步骤进入 `transform(chunk_iterator)` 模式边收边发（parser、chat model 支持）；若某步不支持增量消费，则回退为“聚合完整中间结果再 invoke”——这就是为什么 LCEL 中加不加 parser 不影响打字机效果；
3. **回调贯穿全程**：每步执行时都会基于 `RunnableConfig["callbacks"]` 派生子 run（携带 `parent_run_id`），形成 trace 树 → LangSmith 由此可视化整条链。

---

## 🔧 核心 API 详解

### 表 A：组合构造类

| API | 功能 | 必选参数 | 可选参数 | 注意事项 |
|---|---|---|---|---|
| `\|` 运算符 / `RunnableSequence` | 串行执行 | 各个 `Runnable` | 无 | `__or__/__ror__` 会做强制转换：`str→PromptTemplate`、`Callable→RunnableLambda`、`dict→RunnableParallel` |
| `RunnableParallel` / 字面量 `{...}` 或 `{"key": runnable}` | 并行分支，同一输入喂给所有分支并合并为 dict | 分支名到 Runnable 的映射 | 无 | 短路语义：任一分支失败则整体失败（除非分支内部自带降级） |
| `RunnablePassthrough` | 恒等透传（不做任何事的原样返回） | 无 | 无 | 单独用意义不大，几乎总是配合 `.assign()` |
| `RunnablePassthrough.assign(**fns)` | 在原 dict 上**增量追加**键值 | 键→`Callable(dict)->Any` 映射 | 无 | 多键按定义顺序**依次**求值，后面的 lambda 能读到前面刚写入的键；原键保留 |
| `RunnableLambda(fn)` | 包装普通函数为组件 | 同步函数 `fn` | `afunc=`（显式异步版本）、`name=` | 同步函数在异步调用中会被丢进线程池执行；函数若返回生成器则参与流式 |
| `RunnableBranch(*branches, default)` | 顺序匹配条件路由 | 若干 `(条件fn, Runnable)` 元组 + 最后一个默认分支 Runnable | 无 | 条件按传入顺序短路匹配；不是异步并行评估条件 |
| `RunnableWithFallbacks` | 主路径失败切换备选 | `fallbacks: Sequence[Runnable]` | `exceptions_to_handle` | 由 `.with_fallbacks()` 创建；主备输出类型需一致 |
| `RunnableWithMessageHistory` | 自动注入/持久化历史消息 | `get_session_history` 回调 | `input_messages_key` | 会话记忆的标准做法（替代 `ConversationChain`） |

### 表 B：运行与横切方法（实例方法）

| API | 功能 | 必选参数 | 可选参数 | 注意事项 |
|---|---|---|---|---|
| `.invoke(x, config)` | 单次同步执行 | 输入 | `config: RunnableConfig` | 返回类型由组件决定（str / BaseMessage / dict…） |
| `.batch(inputs, config)` | 并发批量执行 | `list[Input]` | `return_exceptions=False`、`max_concurrency` | 基于 `ThreadPoolExecutor`；`return_exceptions=True` 时错误作为元素混入结果列表 |
| `.stream(x)` / `.astream(x)` | 增量产出 | 输入 | config | 是否真流式取决于链上组件是否实现增量 `transform` |
| `.astream_events(x, version="v2")` | 精细事件流（token/工具/检索粒度） | 输入 | `include_names/tags/types` | 新版中 `version` 参数已默认 v2；用于 UI 打字机 + 过程可视化 |
| `.with_retry(**kwargs)` | 指数退避重试 | 无 | `stop_after_attempt=3`、`retry_if_exception_type=(Exception,)`、`wait_exponential_jitter=True` | 底层是 tenacity；只重试列出的异常类型 |
| `.with_fallbacks([...])` | 失败降级切换 | 备选列表 | `exceptions_to_handle=(Exception,)` | 触发即换备选，**不做退避等待**——重试 vs 降级的本质区别 |
| `.bind(**kwargs)` | 固定传给末端调用的关键字参数 | 无 | 如 `stop=...`、`tools=...`、`tool_choice=...` | 经典用法：ReAct 中 `llm.bind(stop=["\nObservation"])` |
| `.with_config(cfg)` | 固定子链的运行配置 | config dict | 无 | 给子链打 tag/run_name，便于 trace 里区分 |
| `.with_listeners(on_start?, on_end?, on_error?)` | 生命周期钩子 | 无 | 三类回调各一 | 回调签名统一为 `(run: Run, config) -> None` |
| `.pick("a.b")` / `itemgetter` | 从 dict 输入抽取字段 | 键路径 | 无 | 常见于把上游 dict 拆给不同分支 |
| `.configurable_fields(**specs)` | 运行时可切换参数 | 参数名→`ConfigurableFieldSpec` | 无 | 配合 `config={"configurable": {...}}` 实现多租户热切换 |

### 表 C：`RunnableConfig` 关键字段

| 字段 | 作用 | 默认 | 备注 |
|---|---|---|---|
| `callbacks` | 追踪回调（观测） | 由环境注入 | LangSmith 自动挂载就在这里 |
| `tags` / `metadata` | run 标签与结构化元数据 | — | 过滤与分析维度 |
| `run_name` | 该次 run 的显示名 | 类名 | 强烈建议对入口链命名 |
| `max_concurrency` | batch 并发上限 | 无上限（受线程池约束） | 应对供应商 RPM/TPM 限流的钥匙 |
| `recursion_limit` | 图引擎递归深度上限 | 25 | 主要作用于 LangGraph；防无限循环 |
| `configurable` | 动态配置载荷 | — | 配合 `configurable_fields` 使用 |
| `run_id` | 显式指定本次 run ID | 自动 UUID | 用于外部系统关联 |

---

## 💻 实战代码示例

> 统一依赖：`pip install -U langchain langchain-core langchain-openai langchain-community faiss-cpu pydantic python-dotenv pyyaml`
> 根目录 `.env`：`OPENAI_API_KEY=sk-xxx`（可选 `LANGSMITH_TRACING=true`、`LANGSMITH_API_KEY=lsv2-xxx`）

### 场景一：最小组合流水线（LCEL 的 Hello World）

```python
"""scenario1_minimal.py —— 展示最基础的 prompt | llm | parser 三段式组装"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI


def build_chain() -> Runnable[dict, str]:
    """组装链。构造期只创建对象图，不发任何网络请求（惰性求值）。"""
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0,   # 温度归零：答案确定性 => 可复现、利于评测回归
        timeout=30,      # 显式超时，防止推理卡死拖垮服务
        max_retries=2,   # SDK 层幂等重试，应对瞬时网络抖动
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是{domain}领域的资深顾问，回答务必简洁准确。"),
        ("human", "{question}"),
    ])

    # 核心：'|' 通过 __or__ 协议拼接成 RunnableSequence(DAG)，类型随步骤自动推断
    return prompt | llm | StrOutputParser()


if __name__ == "__main__":
    load_dotenv()  # 密钥绝不硬编码，外部注入是生产铁律

    chain = build_chain()

    answer = chain.invoke(
        {"domain": "云计算", "question": "一句话解释冷启动问题"},
        config={
            "run_name": "qa-basic",                 # trace 显示名，排查必备
            "tags": ["demo", "lcel"],               # LangSmith 过滤维度
            "metadata": {"team": "platform", "env": "dev"},
        },
    )
    print(answer)
```

### 场景二：RAG 流水线（并行上下文装配 + 检索为空降级）

```python
"""scenario2_rag.py —— 生产级 RAG：演示 RunnableParallel 双分支装配与证据缺失降级"""

import os
import shutil
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

INDEX_DIR = "./faiss_index"
DOCS = [
    "LangChain 的 LCEL 使用管道符组合组件。",
    "FAISS 是 Meta 开源的高性能向量相似度检索库。",
    "LangSmith 提供 LLM 应用的追踪、评测与监控。",
]


def setup_retriever(k: int = 2):
    """建索引或加载缓存索引（磁盘缓存避免每次重复消耗 embedding 费用）。"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(INDEX_DIR):
        # 安全补丁要求：加载本地 pkl 必须显式确认信任来源
        store = FAISS.load_local(INDEX_DIR, embeddings,
                                 allow_dangerous_deserialization=True)
    else:
        store = FAISS.from_texts(
            DOCS, embeddings,
            metadatas=[{"source": f"doc{i}"} for i in range(len(DOCS))],
        )
        store.save_local(INDEX_DIR)  # 冷启动优化：一次构建多次使用
    return store.as_retriever(search_kwargs={"k": k})


def _format_docs(docs: Sequence[Document]) -> str:
    """渲染检索结果。关键设计：空结果必须显式告知模型，杜绝编造。"""
    if not docs:  # ← 降级点：宁可说没证据，也不让模型幻觉
        return "（未检索到相关资料，请回答'我暂时无法回答'）"
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


def build_rag_chain() -> Runnable[dict, str]:
    retriever = setup_retriever()
    fmt = RunnableLambda(_format_docs, name="format_docs")  # 命名便于 trace 阅读

    llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "严格依据以下资料回答，资料不足就承认不知道。\n\n资料:\n{context}"),
        ("human", "{question}"),
    ])

    # RunnableParallel：同一份输入 {"question": ...} 同时流入两个分支
    rag_body = {
        "context": retriever | fmt,      # 分支A：先取文档列表，再格式化为字符串
        "question": RunnablePassthrough() # 分支B：透传，保证 question 键存活
                     | (lambda x: x["question"]),
    }
    return RunnableParallel(rag_body) | prompt | llm | StrOutputParser()


def safe_ask(chain: Runnable[dict, str], question: str) -> str:
    """业务侧兜底封装：任何上层异常都收敛为用户友好的文案。"""
    try:
        return chain.invoke({"question": question})
    except Exception as exc:  # noqa: BLE001 —— 边界处宽捕获并记录，防止异常裸奔
        print(f"[ERROR] RAG 执行失败: {exc!r}")   # 接日志框架上报告警
        return "服务繁忙，请稍后再试。"             # ← 最终降级文案


if __name__ == "__main__":
    load_dotenv()
    chain = build_rag_chain()
    print(safe_ask(chain, "什么是 LCEL？"))
    print(safe_ask(chain, "量子纠缠怎么做菜？"))  # 触发空检索降级路径
```

### 场景三：结构化抽取（Pydantic 校验驱动重试 + 双模型降级）

```python
"""scenario3_extraction.py —— 工单洞察抽取：校验失败自动重试、主模型挂了换备用模型"""

import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator


class TicketInsight(BaseModel):
    """工单结构化结论。schema 即提示词的一部分（function calling 注入）。"""
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: int = Field(..., ge=1, le=5, description="紧急度 1-5")
    tags: List[str] = Field(default_factory=list, description="最多 5 个标签")

    @field_validator("tags")
    @classmethod
    def cap_tags(cls, v: List[str]) -> List[str]:
        if len(v) > 5:
            raise ValueError("标签数量不得超过 5 个")  # 抛错 => 触发外层重试修正
        return v[:5]


def _build(prompt: ChatPromptTemplate, model: str):
    llm = ChatOpenAI(model=model, temperature=0)
    # ① structured_output：把 Pydantic schema 变成工具强约束，返回值就是模型实例
    extractor = prompt | llm.with_structured_output(TicketInsight)
    # ② 校验循环：验证器抛 ValueError -> with_retry 用更完整输入再试（自愈式重试）
    return extractor.with_retry(stop_after_attempt=2)


def build_extractor() -> Runnable[str, TicketInsight]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "从客服工单中抽取情感、紧急度与标签。"),
        ("human", "{ticket}"),
    ])
    primary = _build(prompt, os.getenv("MODEL_NAME", "gpt-4o"))
    backup = _build(prompt, "gpt-4o-mini")  # 成本/可用性双保险

    route = primary.with_fallbacks(
        [backup],
        # 只有 LLM 相关网络/超时异常才值得换模型重试，其余快速失败
        exceptions_to_handle=(ConnectionError, TimeoutError),
    )
    return route


if __name__ == "__main__":
    load_dotenv()
    chain = build_extractor()
    insight: TicketInsight = chain.invoke(
        "订单三天了还没发货，客服也联系不上，我要投诉并且马上退款！！"
    )
    print(insight.model_dump())  # {'sentiment': 'negative', 'urgency': 5, ...}
```

### 场景四：动态路由（关键词路由 + 子链独立配置）

```python
"""scenario4_router.py —— RunnableBranch：不同意图走不同专家链"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableConfig
from langchain_openai import ChatOpenAI


def make_expert(role: str, run_name: str) -> Runnable[dict, str]:
    """工厂：同构的不同子链，各自携带隔离配置（tag 隔离 trace 归因）。"""
    cfg: RunnableConfig = {"run_name": run_name, "tags": ["router", run_name]}
    prompt = ChatPromptTemplate.from_template(
        f"你是{role}。请用不超过三句话回应：{{question}}"
    )
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0.3)
    return (prompt | llm | StrOutputParser()).with_config(cfg)


def build_router() -> Runnable[dict, str]:
    refund = make_expert("退款专员，熟悉平台退款政策", "expert-refund")
    tech = make_expert("资深技术支持工程师", "expert-tech")
    generic = make_expert("通用智能客服", "expert-generic")

    return RunnableBranch(
        # 条件按顺序短路匹配：命中的第一个分支执行，之后不再评估
        (lambda x: "退款" in x["question"], refund),
        (lambda x: any(w in x["question"] for w in ("报错", "bug", "崩溃")), tech),
        generic,  # 最后位置参数 = 默认分支，永不被跳过
    )


if __name__ == "__main__":
    load_dotenv()
    router = build_router()
    for q in ("我要申请退款", "程序一直崩溃怎么办", "你们几点下班"):
        print(q, "=>", router.invoke({"question": q}), sep="\n  ", end="\n\n")
```

### 场景五：企业级完整脚本（YAML 外置配置 + 熔断器 + Token 监控 + 缓存降级）

**`config.yaml`**

```yaml
model:
  primary: gpt-4o-mini
  fallback: gpt-4o-mini   # 生产建议换成不同供应商（如 Azure 部署），避免同源共障
  temperature: 0
resilience:
  llm_timeout_s: 20
  circuit:
    fail_threshold: 5     # 连续失败 N 次熔断
    reset_timeout_s: 60   # 半开探测间隔
cache:
  enabled: true
  ttl_s: 300
features:
  demo_mode: false
```

**`scenario5_enterprise.py`**

```python
"""scenario5_enterprise.py —— 生产级骨架：配置外置/熔断降级/用量监控/短缓存"""

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI


# ---------- ① 配置外部化：环境变量 + YAML 双通道 ----------
def load_settings(path: Path = Path("config.yaml")) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 敏感信息永远不走 YAML：环境变量注入并校验存在性，fail-fast
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，拒绝启动")
    return cfg


# ---------- ② 简易熔断器（Closed → Open → HalfOpen） ----------
class CircuitBreaker:
    """生产中可替换为 resilience4j / pybreaker；此处展示可内嵌 Runnable 的极简实现。"""

    def __init__(self, fail_threshold: int = 5, reset_timeout: float = 60.0):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self._fails = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()  # 多线程并发计数需要保护

    @property
    def allow(self) -> bool:
        if self._opened_at is None:                        # Closed：放行
            return True
        if time.monotonic() - self._opened_at > self.reset_timeout:  # 半开试探
            return True
        return False                                       # Open：快速失败

    def on_success(self) -> None:
        with self._lock:
            self._fails, self._opened_at = 0, None         # 半开成功 => 回 Closed

    def on_failure(self) -> None:
        with self._lock:
            self._fails += 1
            if self._fails >= self.fail_threshold:
                self._opened_at = time.monotonic()         # 达阈值 => 熔断
                print("[BREAKER] 进入熔断状态")


# ---------- ③ Token 用量观测回调 ----------
class TokenUsageLogger(BaseCallbackHandler):
    """标准化的 usage_metadata 出现于 core>=0.2，跨供应商字段一致。"""

    def __init__(self) -> None:
        self.total = {"input_tokens": 0, "output_tokens": 0}
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for gens in response.generations:
            msg: BaseMessage = gens[0].message
            usage = getattr(msg, "usage_metadata", None) or {}
            with self._lock:  # batch 多线程回调，需加锁累计
                self.total["input_tokens"] += usage.get("input_tokens", 0)
                self.total["output_tokens"] += usage.get("output_tokens", 0)


# ---------- ④ 进程内 TTL 缓存（生产替换为 Redis 语义缓存） ----------
class TTLCache:
    def __init__(self, ttl: float):
        self.ttl, self.store = ttl, {}

    def get(self, key: str) -> Optional[str]:
        hit = self.store.get(key)
        if hit and time.monotonic() < hit[1]:
            return hit[0]
        self.store.pop(key, None)  # 过期清理，防内存泄漏
        return None

    def set(self, key: str, value: str) -> None:
        self.store[key] = (value, time.monotonic() + self.ttl)

    @staticmethod
    def k(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


# ---------- ⑤ 组装带三层防御的业务链 ----------
def build_chain(settings: Dict[str, Any]):
    breaker = CircuitBreaker(**settings["resilience"]["circuit"])
    tokens = TokenUsageLogger()
    cache = TTLCache(settings["cache"]["ttl_s"])

    def guarded_llm_call(payload: dict) -> str:
        q = payload["question"]
        if settings["cache"]["enabled"]:
            if (hit := cache.get(TTLCache.k(q))) is not None:
                return hit                                # L1：命中直接返回
        if not breaker.allow:
            return "系统升级中，请稍后重试。"               # L2：熔断态不再外呼
        try:
            llm = ChatOpenAI(
                model=settings["model"]["primary"],
                temperature=settings["model"]["temperature"],
                timeout=settings["resilience"]["llm_timeout_s"],
                callbacks=[tokens],                       # ③ 观测挂在客户端级
            )
            prompt = ChatPromptTemplate.from_template("简明回答：{question}")
            out = (prompt | llm | RunnableLambda(lambda m: m.content)).invoke(q)
            breaker.on_success()
            if settings["cache"]["enabled"]:
                cache.set(TTLCache.k(q), out)             # L1 写缓存
            return out
        except Exception as exc:                          # noqa: BLE001
            breaker.on_failure()
            print(f"[WARN] 主链路失败，启用兜底话术: {exc!r}")
            return "当前咨询量较大，已为您转人工，请稍候。"  # L3：最终降级话术

    return RunnableLambda(guarded_llm_call, name="enterprise-gateway")


if __name__ == "__main__":
    load_dotenv()
    settings = load_settings()
    chain = build_chain(settings)
    for q in ("LangChain 是什么？", "LangChain 是什么？"):  # 第二次命中缓存
        t0 = time.perf_counter()
        print(chain.invoke({"question": q}), f"({time.perf_counter()-t0:.2f}s)")
```

---

## ⚙️ 配置最佳实践

### 性能调优
1. **批处理优先于循环**：能用 `chain.batch([...], max_concurrency=N)` 就不要串行 `for` 循环 `invoke`；embedding 场景收益尤其大（往返次数 ×N 减少）。`max_concurrency` 是对抗供应商限流的第一开关；
2. **异步入口配异步调用**：FastAPI async 路由里用 `await chain.ainvoke()`；若误用同步 `invoke` 会阻塞整个事件循环；
3. **警惕 RunnableLambda 中的阻塞 IO**：`ainvoke` 时同步函数被丢进默认线程池执行——大量并发会耗尽线程池；高吞吐场景请给 `RunnableLambda(fn, afunc=afn)` 提供原生异步版本；
4. **流式要端到端打通**：中途任何一个同步阻塞的重型 Lambda 都会让下游退化为非流式；
5. **温度、超时、max_retries 显式化**：永远不要依赖库默认值——版本升级时默认值可能变化（隐性行为漂移）。

### 错误处理黄金分层

```text
第0层  请求校验（pydantic 入参模型）
第1层  with_retry          —— 瞬态错误（网络抖动/429），指数退避
第2层  with_fallbacks      —— 持续性错误（模型宕机/区域故障），切换备胎
第3层  业务 try/except     —— 收敛为用户友好文案 + 告警上报
原则：retry 只做秒级忍耐；fallback 不打时间拳（触发即切）；两者可叠加，
     典型结构 = (primary.with_retry()).with_fallbacks([backup.with_retry()])
```

### 日志与监控
- 打开 **LangSmith**：环境变量一键接入（新版用 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`；旧版为 `LANGCHAIN_TRACING_V2`）；入口链必设 `run_name` 与 `tags`；
- 自定义 `BaseCallbackHandler` 统计 token 成本（读标准化的 `usage_metadata`）、P95 延迟、错误率 → 导出到 Prometheus；
- 结构化日志打印 `run_id`，与 trace 系统（LangSmith/Jaeger）双勾稽。

---

## 🕰️ 版本演进

| 版本节点 | 时间 | 与本知识点相关的关键变化 |
|---|---|---|
| v0.0.x | 2023.08 | LCEL 以实验特性发布（"pipe everything" 思想萌芽），`Chain` 类仍是主流 |
| **v0.1** | 2024.01 | 包拆分：核心抽象沉淀进 `langchain-core`，`Runnable`/`RunnableConfig` 定型为公共契约；遗留链开始进入弃用轨道 |
| **v0.2** | 2024.05 | `langchain` 与 `langchain-community` 解耦；`astream_events` v2 成为标准（解决 v1 事件乱序/回调竞态）；官方宣布遗留 Chain 将在后续大版本移除 |
| **v0.3** | 2024.09 | 全面拥抱 **Pydantic v2**（移除 `pydantic.v1` 兼容层依赖路径），自定义 schema 必须用 v2 语法 |
| **v1.0** | 2025.10 | **架构性瘦身**：`langchain` 主包只保留 prompts/models/tools/agents（`create_agent` 底层改由 LangGraph 驱动）；`LLMChain` 等历史类彻底移除（迁至独立的 `langchain-classic` 维护包）；消息内容标准化为 `content_blocks` |

**高频迁移对照表（面试常考）：**

| 遗留 API（已移除/classic 化） | 现代替代方案 |
|---|---|
| `LLMChain(llm, prompt)` | `prompt \| llm \| StrOutputParser()` |
| `SequentialChain` | 直接 `chain_a \| chain_b` 管道串联 |
| `ConversationChain` | `RunnableWithMessageHistory`（进程内记忆）或 LangGraph checkpointer（持久记忆） |
| `RetrievalQA` | `create_retrieval_chain(retriever, doc_chain)`，注意答案键从 `result` 变为 `answer` |
| `ConversationalRetrievalChain` | `history_aware_retriever` + `create_retrieval_chain` 组合 |
| `RouterChain / MultiPromptChain` | `RunnableBranch` 或语义路由（向量相似度选目标链） |
| `AgentExecutor` | LangGraph 预置 `create_agent`（v1.0 起） |

---

## 🆚 替代方案对比

### LCEL vs LlamaIndex Workflows

| 维度 | LangChain LCEL | LlamaIndex Workflow |
|---|---|---|
| 心智模型 | 数据流 DAG（单轮请求内编排） | 事件驱动的有状态工作流 |
| 强项 | 组合原子性强、流式体验成熟、生态集成广 | 数据摄取/Index 体系原生、RAG 开箱即用 |
| 流式传播 | 全链自动传播 chunk | 需在事件间自行传递 stream 对象 |
| 学习曲线 | 低（学会一个协议即可） | 中（事件、ctx、StepWorker 概念多） |
| 互操作 | 二者可互相嵌入对方的组件层（embedder/retriever 均有对方适配） | 同左 |
| 选型经验 | 多模型多工具的对话/Agent 应用 | 文档密集型知识库 + 复杂摄取管线 |

### LCEL vs 裸 Python 编排

自己写 `async def` 串联当然可行，但 LCEL 白送五件套：**流式自动传播、同步/异步自动桥接、一行式重试/降级、结构化 trace 序列化（供 LangServe/LangSmith）、供应商可移植**。手写方案在这五项上都要重复造轮子且容易漏掉边界情况。面试可答：“LCEL 的价值不在语法糖，而在于它把‘运行时合同’标准化了。”

### LCEL vs LangGraph 边界判据

- **单轮请求内、无环、无需共享可变状态** → LCEL；
- **多轮/长任务、存在循环迭代（反思、多 Agent 协作）、需要 checkpoint/HITL 中断恢复** → LangGraph；
- 判断口诀：*“链管一个回合，图管一场战役”*。

---

## 🏢 企业级部署考量

### 并发与容量
- **无状态横向扩展**：LCEL 链本身是无状态纯函数式对象，天然适配 K8s 多副本；但 `RunnableWithMessageHistory` 引入会话粘性需求 → 用 Redis/Postgres 存储后端 + 按用户 ID 分片；
- **三层限流**：入口信号量（保护自身）→ `max_concurrency`（保护下游）→ 令牌桶出站代理（控制成本预算）；
- **截止时间预算**：`asyncio.wait_for(chain.ainvoke(x), timeout=SLO)` 整体兜底 + 组件级 client timeout，双层超时防“组件都在等但没人喊停”。

### 可观测性
- 标准 run tree：一次请求 = 一棵 trace 树（`parent_run_id` 自动级联），LangSmith 按 `tags/team/version` 维度聚合成本看板；
- SLO 建议指标：`ttfb`（首 token 延迟，流式场景专属）、总延迟、`input/output_tokens`、fallback 触发率、缓存命中率；
- **成本护栏**：单请求 token 硬上限 + 月度预算熔断（超限自动切小模型），把成本做成一等公民而非事后账单。

### 安全
- **提示注入防线**：不可信文本（网页抓取/用户上传内容）与指令之间加定界符并声明忽略其中指令；工具授最小权限；高危操作二次人工确认；
- 秘钥管理：仅经环境变量/Secret Manager 注入（见场景五 `fail-fast` 校验）；FAISS 本地反序列化等高危开关必须审计注释留痕；
- PII 合规：上模前经 `RunnableLambda` 做 DLP 脱敏（可接 Presidio），既保护隐私又压缩输入成本。

### 降级策略分级
```text
L0 缓存命中（精确/语义）            → 成本最优，毫秒返回
L1 轻量模型顶替重型模型             → 质量微降，可用性保全
L2 跨供应商 fallback                → 抗单一云厂商故障
L3 熔断 + 静态兜底话术 / 转人工队列   → 保证品牌底线永不崩
每一级都应有独立监控告警，降级率突增 = 事故信号
```

---

## 🎯 高频面试题速答（加分项）

**Q1：为什么 LCEL 能淘汰 LLMChain？**
> LLMChain 把组合固化在类层次里，每种新需求都要继承新类；LCEL 把组合下放为数据（DAG 对象），7 种原语排列组合即可覆盖原有几十个 Chain 类的功能，且流式/异步/重试在协议层一次性解决。这是典型的“类爆炸 → 协议收敛”重构范式。

**Q2：`|` 运算符背后发生了什么？**
> `Runnable.__or__/__ror__` 内部调用 `coerce()`：把右侧对象强制转换——字典转 `RunnableParallel`、函数转 `RunnableLambda`、字符串转 `PromptTemplate`——然后包装为 `RunnableSequence(first, last)`。所以 `chain = "tpl" | fn | llm` 这种混合写法合法。

**Q3：`RunnableSequence.invoke` 和 `.stream` 实现有何区别？**
> invoke 就是 for 循环步步传递；stream 先流式拉取第一步输出，随后尝试让后续步骤以 `transform(chunk_iter)` 模式增量消费，遇到不支持的组件才回退为“攒齐再 invoke”。这也是 `StrOutputParser` 不会打断打字机输出的原因。

**Q4：`RunnablePassthrough.assign` 和 `RunnableParallel` 有何区别？**
> Parallel 把同一输入复制给所有分支、分支之间互不可见、以覆盖方式合并；assign 在单个 dict 上**顺序追加**键，后写的函数能读取先前刚计算出的键，适合“逐步增强上下文”的场景。默认分支不可见性 Parallel 更强的并行度，但 assign 保证了原字典键全部保留。

**Q5：同步函数放进链里，`ainvoke` 会不会阻塞事件循环？**
> 不会——会被丢进线程池执行（因此不会卡 loop 但会占用 worker 线程）；高性能路径应显式提供 `afunc`。另一个坑：lambda 内部调用其他 runnable 时，须在该 runnable 上使用其自身的异步方法，否则事件循环检测会报错或静默退化同步。

**Q6：with_retry 和 with_fallbacks 触发顺序？写成 `(primary.with_retry()).with_fallbacks([backup])` 后 backup 也有 with_retry 会怎样？**
> 先在内层做完所有重试仍失败，才向外触发 fallback；backup 的重试独立执行各自的 N 次。总最坏时长 ≈ (主重试串行和) + (备重试串行和)，工程上必须以此设置总体超时预算。

**Q7：`return_exceptions=True` 时 batch 返回什么？**
> 与输入列表等长的结果列表，失败位置存放的是 Exception 对象而非抛出，需逐元素 isinstance 检查——适合“尽力而为”的离线批量任务。

**Q8：什么时候该放弃 LCEL 改用 LangGraph？**
> 当流程出现：(a) 循环/自我修正迭代；(b) 跨请求持久状态；(c) 人审中断-恢复；(d) 多 Agent 控制权移交。任一成立即上 Graph；仅单轮线性变换继续用 LCEL 保持轻量。

---

## ✅ 小结 · 一页带走

- **一个协议**：`Runnable(invoke/batch/stream + a*)` 定义运行时合同；
- **一种语法**：`|` 组合一切，构造期建 DAG、惰性执行；
- **两把伞**：`with_retry`（瞬态退避）、`with_fallbacks`（持续切换）；
- **三个深挖点**：Sequence 的 stream 回退机制、assign 的顺序合并语义、sync↔async 线程池桥接；
- **边界心法**：链管一个回合，图管一场战役。

---

如果你想把这份手册模板套用到**其他知识点**（例如：Memory 与 RunnableWithMessageHistory、Tool Calling 与 Agent 内核、Output Parser 家族、Document Loader/Splitter 管线、LangGraph 状态机……），回复知识点名称，我按同样深度重新展开。
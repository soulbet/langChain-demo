# LangChain Agent Middleware 全部 Hooks 总结

> 目录：`ai_app/LangChain/middleware`
> 版本基线：langchain **1.3.14** / langchain-core **1.4.9**（本机 `ai-env`）
> 配套代码：[`middleware_demo.py`](middleware_demo.py)

---

## 1. 什么是 Middleware

LangChain 1.x 的 **Agent 中间件** 允许你在 agent 主循环的关键节点上插入自定义逻辑，从而在不改动 agent 结构的前提下实现：动态提示词、重试、回退、限流、PII 脱敏、日志监控等。

使用方式是把中间件实例传给 `create_agent`：

```python
from langchain.agents import create_agent

agent = create_agent(model, tools=[...], middleware=[MyMiddleware(), OtherMiddleware()])
```

- 主循环大致是：`before_agent → [before_model → model → after_model → (tools) → ...] → after_agent`
- **中间件有顺序**：列表越靠前、越"外层"（first = outermost）。
- 每个 hook 都有 **同步 / 异步** 两个版本；用 `invoke/stream` 走同步，用 `ainvoke/astream` 走异步。

---

## 2. 全部 Hooks 一览

| Hook | 触发时机 | 典型用途 | 返回值 |
|------|---------|---------|--------|
| `before_agent` / `abefore_agent` | agent 开始执行前（整个 run 一次） | 初始化、鉴权、注入上下文 | `dict \| None`（更新 state） |
| `before_model` / `abefore_model` | **每次**调用模型前 | 注入状态、动态日志、并发控制 | `dict \| None` |
| `wrap_model_call` / `awrap_model_call` | 包裹模型调用（可拦截/重试/短路） | 重试、降级、缓存、改写请求/响应 | `ModelResponse \| AIMessage \| ExtendedModelResponse` |
| `after_model` / `aafter_model` | **每次**模型响应后 | 记录响应、控制是否继续 | `dict \| None` |
| `wrap_tool_call` / `awrap_tool_call` | 包裹工具调用 | 工具重试、监控、参数改写 | `ToolMessage \| Command` |
| `after_agent` / `aafter_agent` | agent 执行结束（整个 run 一次） | 汇总、上报、清理 | `dict \| None` |
| `dynamic_prompt` | 即 `wrap_model_call` 的专用封装 | 动态生成 system 提示词 | `str \| SystemMessage` |

> 说明：`before_model` / `after_model` / `before_agent` / `after_agent` 的返回值也可以是
> `Command`，用于跳转（配合 `hook_config(can_jump_to=...)`）。

---

## 3. Hook 签名详解

所有方法都接收 `state`（agent 状态）和 `runtime`（运行时上下文）。

### 3.1 `before_agent`
```python
def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
```

### 3.2 `before_model`
```python
def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
```

### 3.3 `wrap_model_call`（重点）
```python
def wrap_model_call(
    self,
    request: ModelRequest,                                             # 含 model/messages/tools/state/runtime
    handler: Callable[[ModelRequest], ModelResponse],                  # 调用它以真正执行模型
) -> ModelResponse | AIMessage | ExtendedModelResponse: ...

async def awrap_model_call(self, request: ModelRequest,
                           handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse | AIMessage | ExtendedModelResponse: ...
```
- 通过 `handler(request)` 执行模型；**可多次调用实现重试**，也可**不调用直接短路**返回一个结果。
- `request.override(...)` 可生成新的请求（改 `model` / `system_message` / `messages` / `tools` 等）。
- 可返回 `ExtendedModelResponse(model_response=..., command=Command(...))` 在模型节点后追加状态更新。

### 3.4 `after_model`
```python
def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
```

### 3.5 `wrap_tool_call`
```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,        # 含 tool_call dict / BaseTool / state / runtime
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command: ...

async def awrap_tool_call(self, request: ToolCallRequest,
                          handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command: ...
```
- 通过 `handler(request)` 执行工具；可多次调用实现重试，或改写 `request.tool_call["args"]`。
- `request.override(tool_call=...)` 生成修改后的调用。

### 3.6 `after_agent`
```python
def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None: ...
```

### 3.7 `dynamic_prompt`
```python
@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:      # 返回提示词字符串
    return f"你是助手，帮 {request.runtime.context.get('user', 'user')} 解决问题。"
```
本质是 `wrap_model_call` 的便捷封装，用于按状态动态生成 system prompt。

---

## 4. 类属性（AgentMiddleware）

| 属性 | 说明 |
|------|------|
| `state_schema` | 自定义 state schema（默认 `AgentState`；可加自定义字段） |
| `tools` | 中间件额外注册的工具（会并入 agent 工具集） |
| `transformers` | 流式 transformer 工厂序列，用于自定义 `stream_mode="messages"` 的转换 |
| `name` | 中间件名字（默认类名；用于图节点命名、去重校验） |

---

## 5. `hook_config` 与条件跳转

`@hook_config(can_jump_to=[...])` 给 `before_model` / `after_model`（或对应装饰器）添加跳转能力，
在返回值里用 `{"jump_to": "..."}` 改变执行流向：

```python
@hook_config(can_jump_to=["end", "model", "tools"])
def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if some_condition(state):
        return {"jump_to": "end"}     # 直接结束
    return None
```

合法目标：`"tools"`、`"model"`、`"end"`。

---

## 6. 第二种写法：装饰器（无需子类化）

```python
from langchain.agents.middleware import (
    before_agent, before_model, after_model, after_agent,
    wrap_model_call, wrap_tool_call, dynamic_prompt,
)

@before_model
def log(state, runtime):
    print("about to call model")

@dynamic_prompt
def prompt(request):
    return "你是测试助手。"

agent = create_agent(model, middleware=[log, prompt])
```

适合"只关心单个 hook"的轻量场景。

---

## 7. 生命周期顺序（实测输出）

运行 `middleware_demo.py` 得到的 hook 顺序如下（一轮：模型出工具调用 → 执行工具 → 再调模型 → 结束）：

```
before_agent
before_model
wrap_model_call        → 模型返回（带 tool_calls）
after_model
wrap_tool_call         → 工具执行
before_model           ← 回到循环
wrap_model_call        → 模型返回（最终答案）
after_model
after_agent
```

即：**`before_agent` 与 `after_agent` 各跑一次；`before_model / wrap_model_call / after_model` 每轮模型调用各跑一次；`wrap_tool_call` 每次工具调用跑一次。**

---

## 8. 内置中间件（开箱即用）

`langchain.agents.middleware` 还提供可直接使用的中间件：

| 名字 | 作用 |
|------|------|
| `ModelRetryMiddleware` | 模型调用重试 |
| `ModelFallbackMiddleware` | 模型回退（主模型失败换备用模型） |
| `ModelCallLimitMiddleware` | 限制模型调用次数 |
| `ToolRetryMiddleware` | 工具调用重试 |
| `ToolCallLimitMiddleware` | 限制工具调用次数 |
| `ToolErrorMiddleware` | 统一处理工具错误 |
| `ToolSelectionMiddleware` | 工具选择 |
| `SummarizationMiddleware` | 长对话/长下文摘要 |
| `TodoListMiddleware` | 任务拆解（To-Do list） |
| `HumanInTheLoopMiddleware` | 人工介入 |
| `PIIMiddleware` | 脱敏 |
| `ContextEditingMiddleware` | 上下文编辑 |
| `FilesystemFileSearchMiddleware` | 文件系统检索 |
| `ProviderToolSearchMiddleware` | 外部工具检索 |
| `ShellToolMiddleware` | shell 工具及沙箱策略 |

---

## 9. 在本项目中使用

`create_model` 走项目 `ModelFactory`，把中间件直接挂到 `create_agent`：

```python
from ai_app.model_factory.model_factory import ModelFactory
from langchain.agents import create_agent
from ai_app.agent_demo.middleware.middleware_demo import DemoMiddleware, my_dynamic_prompt

llm = ModelFactory().create_model()  # 本地/云模型
agent = create_agent(
        llm,
        tools=[...],
        middleware=[DemoMiddleware(), my_dynamic_prompt],
)
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
```

---

## 10. 常见坑位

- **同步/异步要匹配**：只写了 `awrap_model_call` 但用 `invoke()` 触发会抛 `NotImplementedError`（反之亦然）。保持 hook 的 sync/async 版本与调用方式一致。
- **`bind_tools` 缺失**：测试用的伪模型需实现 `bind_tools`（见 `BindableFakeChatModel` 示例），否则 `create_agent` 会报 `NotImplementedError`。
- **中间件顺序**：列表越靠前越"外层"，`wrap_model_call`/`wrap_tool_call` 的"最外层"先执行。
- **同一 name 去重**：`create_agent` 会校验中间件 `name` 不能重复。

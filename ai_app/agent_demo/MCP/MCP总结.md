# MCP（Model Context Protocol）学习总结

> 目录：`ai_app/LangChain/MCP`
> 说明：基于本目录下 3 个演示文件整理的 MCP 使用总结。
> 依赖库：`langchain-mcp-adapters`、`langchain`、`mcp`/`fastmcp`、`elasticsearch-mcp`。

---

## 1. MCP 是什么

**Model Context Protocol（模型上下文协议）** 是 Anyscale 提出的一套开放协议，用于让大模型（LLM/Agent）以**统一的方式**接入外部工具、数据源和 API。

核心思想：
- **MCP Server**：把工具（tools）、资源、提示词暴露成标准接口，可以运行在本地子进程（`stdio`）或远程 HTTP（`http`/`sse`）。
- **MCP Client**：由应用（如 LangChain Agent）连接一个或多个 Server，把暴露的工具变成普通工具给模型调用。
- 好处：工具"即插即用"，多个 MCP Server 可以自由组合，无需为每个数据源手写对接逻辑。

对应到本项目三种典型角色：
- **Provider / Server 端**：`demo02-custom_servers.py` 用 `FastMCP` 自定义一个 数学 Server。
- **Client 端**：`demo01.py`、`demo03-es_query.py` 用 `MultiServerMCPClient` 连接 Server 并获取工具。
- **编排端**：用 `langchain` 的 `create_agent` 把这些工具挂到 Agent 上，让 LLM 自动选择调用。

---

## 2. 目录文件一览

| 文件 | 作用 | 关键库 |
|------|------|--------|
| `demo01.py` | MCP 客户端：同时连接 `stdio` + `http` 两个 Server，创建 Agent 问答 | `langchain_mcp_adapters`、`langchain` |
| `demo02-custom_servers.py` | 自定义 MCP Server：用 FastMCP 暴露 `add` / `multiply` 工具 | `fastmcp` |
| `demo03-es_query.py` | 接入 es-mcp-server，用自然语言查询 Elasticsearch | `langchain_mcp_adapters`、`es-mcp-server` |
| `__init__.py` | 空文件，标识为一个 Python 包 | — |

---

## 3. demo01.py — MCP 客户端基础（多 Server 连接）

**核心代码：**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "math": {                                # 本地子进程 Server（stdio）
            "transport": "stdio",
            "command": "python",
            "args": ["/path/to/math_server.py"],
        },
        "weather": {                             # 远程 HTTP Server
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        }
    },
    handle_tool_errors=False,                    # 工具抛错时直接抛异常
)

llm = ModelFactory().create_model()
tools = await client.get_tools()                 # 汇总所有 Server 的工具
agent = create_agent(llm, tools)
response = await agent.ainvoke({"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]})
```

**要点：**
- `MultiServerMCPClient` 接受一个 **dict**，key 是 Server 名，value 是连接配置，可同时持有多类 Server。
- 两种 transport：
  - `stdio`：通过命令拉起本地子进程（本地、私密、无需网络）。
  - `http`：连接远程已部署的 Server（跨机器、可共用）。
- `handle_tool_errors=False`：工具执行出错时把异常抛出来便于定位（默认 True 会吞掉错误）。
- `await client.get_tools()`：一次性拿到所有 Server 的工具，交给 `create_agent` 即可。
- 模型通过 `ModelFactory().create_model()` 创建（本项目自定义模型工厂，默认走本地 `http://localhost:55002/v1`）。

**运行前提：** `math_server.py` 需存在，且 `weather` HTTP Server 需已在 `:8000` 启动。

---

## 4. demo02-custom_servers.py — 自定义 MCP Server

**核心代码：**

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")   # 以 stdio 模式启动
```

**要点：**
- 用 `FastMCP("名字")` 建一个 Server 实例。
- 用 `@mcp.tool()` 装饰一个普通 Python 函数，函数签名 + 类型注解会自动生成工具 schema，其 docstring 会作为描述给模型看。
- `mcp.run(transport="stdio")`：以本地子进程协议运行，供 `MultiServerMCPClient` 以 `command="python"` 的方式拉起（对应 demo01 的 `math` Server）。
- 这就是"自定义 Server"的最小闭环，可把任意 Python 函数暴露成 MCP 工具。

---

## 5. demo03-es_query.py — 接入 es-mcp-server 做自然语言查询

**核心代码：**

```python
client = MultiServerMCPClient({
    "elasticsearch": {
        "transport": "stdio",
        "command": "wsl",
        "args": ["-d", "Ubuntu", "bash", "-lc",
                 "export ES_HOST=localhost && export ES_PORT=9201 && "
                 "/home/zyfu/.local/bin/uvx es-mcp-server --es-version 8"]
    }
})

tools = await client.get_tools()
print("发现工具:", [tool.name for tool in tools])

llm = model_factory().create_model()      # 注意：demo01 是 ModelFactory().create_model()
agent = create_agent(llm, tools, system_prompt="""你是一个执行助手... 直接返回查询结果""")

response = await agent.ainvoke({"messages": [{"role": "user", "content": "列出所有索引"}]})
```

**要点：**
- Es 查询是通过 **WSL + uvx** 拉起 `es-mcp-server`（`--es-version 8`），连接 `localhost:9201` 的 ES。
- MCP 工具集里含如 `list_indices`、`search` 等；代码里通过 `[tool.name for tool in tools]` 打印发现的工具。
- 用 `system_prompt` 约束 Agent：**直接调用工具执行、不要输出执行计划、严格按指定 JSON 结构构建查询**，再用 `search` 工具执行。
- 用自然语言发起查询（列出所有索引 / 搜索 `user_info` 索引里 `name=zhangsan` 的 `age`）。
- `response["messages"][-1].content` 取最终回答；遍历 `messages` 可打印工具调用（`tool_calls`）与结果。

> ⚠️ 注意：demo03 里 `model_factory().create_model()` 用的是**小写函数** `model_factory`，
> 而 demo01 用的是 `ModelFactory` **类**。当前 `model_factory.py` 只定义了 `ModelFactory` 类，
> 并未定义 `model_factory` 函数，因此 demo03 的 import 需确认是否正确（可能应改为 `ModelFactory().create_model()`）。

---

## 6. 关键概念与流程

```
[自定义 Server：FastMCP]      [第三方 Server：es-mcp-server]
        │                              │
        │  stdio / http                │  stdio (WSL+uvx)
        ▼                              ▼
        └────────── MultiServerMCPClient ──────────┘
                          │
                          │ await client.get_tools()
                          ▼
                     tools（统一工具列表）
                          │
                          ▼
                 create_agent(llm, tools)
                          │
                          ▼
              Agent 按需自动调用工具并返回结果
```

**要记住的点：**
1. **Client ↔ Server 两端分离**：Server 只负责"暴露工具"，Client 只负责"连接并取工具"，模型只负责"决定调哪个工具"。
2. **transport 两种模式**：`stdio`（本地进程）与 `http`（远程服务）；一个 client 可混用。
3. **工具的发现与命名**：Server 里每个 `@mcp.tool()` / 三方 Server 暴露的工具都会变成纯工具，交给 langchain 的 agent。
4. **提示词规范很重要**：es-mcp 这类工具需要 LLM 构造特定 JSON，系统提示词里给出模板能显著提升命中率。

---

## 7. 环境与依赖

| 库 | 用途 |
|----|------|
| `langchain-mcp-adapters` | MCP ↔ LangChain 适配，提供 `MultiServerMCPClient` |
| `langchain` | `create_agent` 构建 Agent |
| `mcp` | MCP 协议的 Python 实现（底层） |
| `fastmcp` | 快速定义自定义 MCP Server |
| `elasticsearch-mcp` | 外部 ES MCP Server（经 `uvx es-mcp-server` 运行） |
| `elasticsearch` | ES 客户端/查询（demo03 依赖） |

模型统一走项目的 `ModelFactory`（默认本地模型，`http://localhost:55002/v1`）。

---

## 8. 常见坑位

- **transport 不匹配**：Client 与 Server 必须一致——Server 用 `stdio` 启动，Client 就要配 `transport: "stdio"` + `command`/`args`。
- **路径 / 端口写死**：demo01 的 `math_server.py` 路径、weather 的 `:8000`、demo03 的 `local/9201` 都需要按实际改。
- **模型来源不一致**：demo01 与 demo03 用了不同的模型创建写法（见上），需统一。
- **日志刷屏**：demo03 在顶部把 `elastic_transport`、`mcp` 日志降到 `WARNING` 以抑制 INFO 输出。

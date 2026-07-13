import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from model_factory import model_factory
import logging

# 在代码顶部添加，抑制 elastic_transport 的 INFO 日志
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)


async def main():
    # 1. 配置 MCP 客户端，连接你本地启动的 es-mcp-server
    # 注意：你的服务器用 stdio 模式启动，所以这里 transport 要匹配
    client = MultiServerMCPClient(
        {
            "elasticsearch": {
                "transport": "stdio",
                "command": "wsl",
                "args": [
                    "-d",
                    "Ubuntu",
                    "bash",
                    "-lc",
                    "export ES_HOST=localhost && "
                    "export ES_PORT=9201 && "
                    "/home/zyfu/.local/bin/uvx "
                    "es-mcp-server "
                    "--es-version 8"
                ]
            }
        }
    )

    # 2. 获取所有 MCP 工具（例如 list_indices, search_docs 等）
    tools = await client.get_tools()

    print("发现工具:", [tool.name for tool in tools])

    # 3. 创建 Agent（使用你的本地 Qwen 模型）
    llm = model_factory().create_model()
    agent = create_agent(llm, tools,
                         system_prompt="""
                         你是一个执行助手。当用户提出查询请求时，请直接调用工具执行查询，不要输出执行计划或步骤描述。直接返回查询结果。
                         于查询 name=zhangsan：
                            {
                                "index": "user_info",
                                "queryBody": {
                                    "query": {
                                        "match": {
                                            "name": "zhangsan"
                                        }
                                    }
                                }
                            }

                            请严格按照此格式构建查询参数，不要简化或修改结构，并使用`search`工具执行查询语句。
                         """)

    # 4. 自然语言查询 ES
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "列出所有索引"}]
    })
    print(response["messages"][-1].content)

    # 5. 也可以搜索文档
    response2 = await agent.ainvoke({
        "messages": [{"role": "user",
                      "content": "使用查询语句，在 user_info 索引中搜索{'name':zhangsan}的年龄,年龄字段名为 `age`，返回张三的年龄"}]
    })
    # 打印完整响应
    for msg in response2["messages"]:
        print(f"--- {msg.__class__.__name__} ---")
        print(msg.content if msg.content else msg.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())

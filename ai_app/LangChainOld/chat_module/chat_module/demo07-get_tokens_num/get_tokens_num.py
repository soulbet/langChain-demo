from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from ai_app.model_factory.model_factory import model_factory

llm=model_factory().create_model()
dp_response = llm.invoke("hello")
print(dp_response.usage_metadata)
print(dp_response.response_metadata)

# 流
aggregate = None
for chunk in llm.stream("hello", stream_usage=True):
    print(chunk)
    aggregate = chunk if aggregate is None else aggregate + chunk
print(aggregate)
print(aggregate.content)
print(aggregate.usage_metadata)

""" get_openai_callback 追踪和记录 LLM 调用的成本与 Token 消耗
用于监控、分析和成本控制
作为一个上下文管理器（with 语句），它会自动收集其代码块内所有 LLM 调用的 Token 使用量
"""
with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    print(f"总 Token 数: {cb.total_tokens}")  # Prompt + Completion 的总 Token 数
    print(f"输入 Token 数: {cb.prompt_tokens}")  # Prompt 消耗的 Token
    print(f"输出 Token 数: {cb.completion_tokens}")  # 生成内容消耗的 Token
""" 
create_tool_calling_agent 构建智能体，让 LLM 能够自主调用外部工具或函数
用于构建能执行任务、获取实时信息的复杂 AI 应用
将定义了名称、描述和输入参数的工具（Tools）绑定给一个支持函数调用（Function Calling）的大模型
"""
# 1. 定义一个工具
@tool
def multiply(x: int, y: int) -> int:
    """计算两个整数的乘积。"""
    return x * y
# 3. 创建提示模板，必要的占位符是 agent_scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以使用你的工具。"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. 创建 Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=[multiply], # 将工具传给 Agent
    prompt=prompt
)

# 5. 创建 Agent 执行器，它会自动处理调用循环
agent_executor = AgentExecutor(agent=agent, tools=[multiply], verbose=True)

with get_openai_callback() as cb:
    response = agent_executor.invoke({"input": "请问 25 乘以 4 等于多少？"})
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
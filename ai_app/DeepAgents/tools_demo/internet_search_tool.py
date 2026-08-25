from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ai_app.DeepAgents.tools_demo.custom_real_time_tool import SearchTool
from ai_app.model_factory.model_factory import ModelFactory



search_tool = SearchTool()
llm = ModelFactory().create_model(local_model_type="coder")

agent = create_agent(
    model=llm,
    tools=[search_tool]
)


"""2、然后使用工具 search_tool 查今天及未来几天的上海天气怎么样
输出格式：日期+天气情况+未来几天的日期"""
result = agent.invoke({"messages":[HumanMessage(content="""
请**直接**调用 search_tool 工具来回答以下问题，**不要输出任何计划、解释或思考过程**：
我所在的城市是 上海
1、查一下今天的日期
2、查一下今天的天气
3、查一下明天的天气
""")]})


last_message = result["messages"][-1]
# 如果是 AIMessage 对象，使用 .content
if hasattr(last_message, "content"):
    content = last_message.content
    print(content)

total_input = 0
total_output = 0

for msg in result.get("messages", []):
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        total_input += msg.usage_metadata.get("input_tokens", 0)
        total_output += msg.usage_metadata.get("output_tokens", 0)

print(f"总输入 tokens: {total_input}")
print(f"总输出 tokens: {total_output}")

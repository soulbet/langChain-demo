from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from model_factory import model_factory


# 1. 定义一个示例工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气。"""
    # 模拟API调用
    return f"{city}的天气是晴朗的，温度25°C。"


# 2. 初始化模型并绑定工具
llm = model_factory().create_model()
llm_with_tools = llm.bind_tools([get_weather])

# 3. 用户提问
user_query = "上海今天天气怎么样？"
messages = [HumanMessage(content=user_query)]


# 4. 第一轮：模型决定调用工具
response = llm_with_tools.invoke(messages)
messages.append(response)  # 将AIMessage加入历史

# 5. 检查并执行工具调用
if response.tool_calls:
    # 假设只处理第一个工具调用
    tool_call = response.tool_calls[0]
    if tool_call["name"] == "get_weather":
        # 执行工具
        tool_result = get_weather.invoke(tool_call["args"])

        # 6. 创建ToolMessage并添加到历史
        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]  # 关键：关联请求和响应
        )
        messages.append(tool_message)

        # 7. 第二轮：将工具结果发给模型，生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(final_response.content)
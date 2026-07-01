import operator
from typing import TypedDict, Annotated, Literal

from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from main.model_factory import model_factory

llm = model_factory().create_model()


# 定义工具和模型
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# 绑定参数
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

# 图的状态用于存储消息和LLM调用次数。
# 通过 Annotated[list, operator.add] 的方式，确保所有消息（包括用户输入、LLM 思考、工具结果）都被有序地追加到对话历史中，供后续节点使用
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# 定义模型节点
# 是智能体的“大脑”，将用户消息和系统提示发送给绑定了工具的 LLM，让模型判断下一步是调用工具还是直接回答
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# 定义工具节点
# 是智能体的“手脚”。负责解析 LLM 返回的 tool_calls 指令，并实际运行对应的工具函数，最后将执行结果封装成 ToolMessage 返回
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

#  定义终端逻辑
# 是智能体的“方向盘”。检查最后一条消息是否有工具调用请求，如果有，则路由到 tool_node 执行；否则结束流程，返回最终答案
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# 构建工作流
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue, # 根据当前的状态决定，决定下一个执行的节点
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# 编译agent
agent = agent_builder.compile()

# 展示agent
from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
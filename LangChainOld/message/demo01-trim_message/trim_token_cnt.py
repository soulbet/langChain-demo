from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, trim_messages
)
from langchain_core.messages.utils import count_tokens_approximately

"""
trim_messages 函数的参数：
max_tokens	         修剪后消息允许的最大 token 数或消息条数。
token_counter	     最关键。用于计算 token 数量的函数。可以是具体的模型实例（如 ChatOpenAI）、字符串 'approximate'（近似计数），或自定义函数（如 len 用于按消息条数计数）。
strategy	         修剪策略。'last'（保留最近的消息，常用）或 'first'（保留最早的消息）。
include_system	     是否始终保留第一条系统消息（SystemMessage），通常设为 True。
start_on / end_on	 确保修剪后的消息以特定类型（如 HumanMessage）开始或结束，维持对话结构有效。
allow_partial	     是否允许拆分单条消息以精确达到 max_tokens，设为 True 可能截断不完整消息。

"""

messages = [
    SystemMessage("你是一个乐于助人的助手。"),
    HumanMessage("嗨！"),
    AIMessage("你好！有什么可以帮你的？"),
    HumanMessage("LangChain是什么？"),
    AIMessage("LangChain是一个用于开发由语言模型驱动的应用程序的框架。"),
    HumanMessage("给我简单介绍一下。"),
]

# 修剪消息，保留约45个token
trimmed = trim_messages(
    messages,
    max_tokens=45,  # 修剪后消息允许的最大 token 数或消息条数。
    strategy="last",  # 保留最近的消息
    token_counter=count_tokens_approximately,  # 使用近似计数器
    include_system=True,  # 保留系统消息
    start_on="human",  # 确保以HumanMessage开头
)

print(trimmed)
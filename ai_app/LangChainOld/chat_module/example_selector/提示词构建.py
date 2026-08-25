from langchain_core.prompts import ChatPromptTemplate

"""
from_messages /from_template = 建模板
format_messages /format_prompt/invoke = 填变量、生成最终提示
"""

# 返回 list[BaseMessage]
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 helpful 助手"),
    ("human", "讲一个关于 {topic} 的笑话")
])
# 仅 1.x 可用
prompt1 = ChatPromptTemplate.from_template("""
system: 你是一个 helpful 助手
user: 讲一个关于 {topic} 的笑话
""")
# 返回 ChatPromptValue
pv = prompt.format_prompt(topic="猫")
# 可以转字符串或转消息：
print(pv.to_string())
print(pv.to_messages())
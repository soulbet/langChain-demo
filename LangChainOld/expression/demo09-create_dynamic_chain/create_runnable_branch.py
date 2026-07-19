
"""
指的是根据运行时的输入或条件，动态地决定链的执行路径或结构

"""
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

# 定义两个模板：一个用于代码，一个用于其他
code_template = PromptTemplate.from_template("用 Python 编写一段代码：{input}")
general_template = PromptTemplate.from_template("回答问题：{input}")

# RunnableBranch 是构建动态链的基础组件，它的核心是提供 “if-elif-else” 逻辑，允许你在运行时根据条件选择执行哪一段链
# 构建分支链
dynamic_chain = RunnableBranch(
    (lambda x: "python" in x["input"].lower(), code_template | model), # 条件1
    (lambda x: "数学" in x["input"], general_template | model), # 条件2
    general_template | model  # 默认分支
)

# 测试
print(dynamic_chain.invoke({"input": "用 python 计算斐波那契数列"})) # 触发条件1
print(dynamic_chain.invoke({"input": "解方程 x^2=4"}))               # 触发条件2
print(dynamic_chain.invoke({"input": "你好"}))                        # 触发默认

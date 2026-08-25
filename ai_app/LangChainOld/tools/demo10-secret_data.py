from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

@tool
def foo(x: int, config: RunnableConfig) -> int:
    # 从配置中读取机密信息
    secret = config["configurable"]["__top_secret_int"]
    return x + secret

# 在调用时传入机密信息
result = foo.invoke(
    {"x": 5},
    {"configurable": {"__top_secret_int": 2, "traced_key": "bar"}}
)
print(result)  # 输出: 7
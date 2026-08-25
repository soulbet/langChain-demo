from langchain_core.runnables import RunnableLambda, RunnableFallback
from langchain_core.exceptions import OutputParserException
import httpx

def risky_operation(input_dict):
    """可能失败的主操作"""
    # 模拟可能失败的场景
    if input_dict.get("should_fail"):
        raise ConnectionError("API 连接失败")
    return f"处理结果: {input_dict['data']}"

def fallback_operation(input_dict):
    """后备操作"""
    return f"后备结果: {input_dict['data']} (使用缓存数据)"

# 创建可运行对象
primary_runnable = RunnableLambda(risky_operation)
fallback_runnable = RunnableLambda(fallback_operation)

# 添加后备方案（指定捕获的异常类型）
robust_runnable = primary_runnable.with_fallbacks(
    [fallback_runnable],
    exception_types=(ConnectionError, TimeoutError, httpx.ConnectError)
)

# 测试
print(robust_runnable.invoke({"data": "test", "should_fail": False}))  # 使用主方案
print(robust_runnable.invoke({"data": "test", "should_fail": True}))   # 触发后备方案
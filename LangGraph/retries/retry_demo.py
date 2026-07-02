from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph
import requests

# 示例1：仅重试网络相关异常，最多尝试5次
retry_policy = RetryPolicy(
    max_attempts=5,
    retry_on=(requests.ConnectionError, requests.Timeout)
)

# 示例2：使用函数自定义重试条件
def should_retry(error: Exception) -> bool:
    # 只重试特定HTTP状态码
    if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        return error.response.status_code in [429, 502, 503, 504]
    return False

retry_policy = RetryPolicy(
    max_attempts=3,
    retry_on=should_retry,
    initial_interval=1.0,
    max_interval=60.0
)

# 应用到节点
graph_builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=retry_policy
)
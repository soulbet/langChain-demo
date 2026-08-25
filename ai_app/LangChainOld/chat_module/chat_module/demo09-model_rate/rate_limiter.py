import time

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 超慢！我们每10秒只能请求一次！！
    check_every_n_seconds=0.1,  # 每100毫秒唤醒一次，检查是否允许发出请求
    max_bucket_size=10,  # 控制最大突发数据量。
)
model = ChatOpenAI(
            model="deepseek-chat",  # DeepSeek 模型名称
            openai_api_key="sk-83d73e8bca6948159a4f2d2b6d9abf94",  # 👈 DeepSeek 的 API Key
            openai_api_base="https://api.deepseek.com/v1",  # 👈 DeepSeek 的 API 端点
            temperature=0.7,
            max_tokens=1024,
            logprobs=True,  # 开启 logprobs
            top_logprobs=5,  # 5个备选
            rate_limiter=rate_limiter
        )
for _ in range(5):
    tic = time.time()
    model.invoke("hello")
    toc = time.time()
    print(toc - tic)
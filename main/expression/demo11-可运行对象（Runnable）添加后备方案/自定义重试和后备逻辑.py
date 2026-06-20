from langchain_core.runnables import RunnableLambda
from tenacity import retry, stop_after_attempt, retry_if_exception_type


class RobustRunnable:
    """带重试和后备方案的可运行对象"""

    def __init__(self, primary, fallbacks=None, max_retries=3):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _invoke_with_retry(self, input_data):
        return self.primary.invoke(input_data)

    def invoke(self, input_data):
        try:
            # 先尝试主方案（带重试）
            return self._invoke_with_retry(input_data)
        except Exception as e:
            print(f"主方案失败: {e}")
            # 依次尝试后备方案
            for i, fallback in enumerate(self.fallbacks, 1):
                try:
                    print(f"尝试后备方案 {i}")
                    return fallback.invoke(input_data)
                except Exception as fb_e:
                    print(f"后备方案 {i} 失败: {fb_e}")
                    continue
            # 所有方案都失败
            raise RuntimeError("所有方案都失败了")


# 使用示例
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

primary = PromptTemplate.from_template("处理: {input}") | ChatOllama(model="deepseek-r1:7b")
fallback1 = PromptTemplate.from_template("备用处理: {input}") | ChatOllama(model="llama3.2:3b")
fallback2 = lambda x: f"离线处理: {x['input']}"

robust = RobustRunnable(primary, [fallback1, fallback2])
result = robust.invoke({"input": "test"})
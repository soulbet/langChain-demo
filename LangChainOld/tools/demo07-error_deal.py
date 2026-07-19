
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# 主模型和备选模型
primary_model = ChatOpenAI(model="gpt-4")
fallback_model = ChatAnthropic(model="claude-3-sonnet-20240229")

# 使用 fallback 方法：如果主模型调用失败，自动切换到备选模型
chain = primary_model | (lambda x: x.content) | SomeParser()
safe_chain = chain.fallback(fallback=fallback_model)

# 调用时，如果主模型出错，会无缝切换至 Claude
result = safe_chain.invoke("Hello, world!")
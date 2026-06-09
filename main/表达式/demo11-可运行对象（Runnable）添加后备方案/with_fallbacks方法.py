from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 主方案：使用本地 Ollama
primary_llm = ChatOllama(model="deepseek-r1:7b", base_url="http://localhost:11434")

# 后备方案1：使用其他本地模型
fallback_llm_1 = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")

# 后备方案2：使用 OpenAI（如果配置了 API Key）
fallback_llm_2 = ChatOpenAI(model="gpt-3.5-turbo")

# 添加后备方案（按顺序尝试）
robust_llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2])

# 使用带后备的模型
chain = robust_llm | StrOutputParser()
result = chain.invoke("Hello, how are you?")
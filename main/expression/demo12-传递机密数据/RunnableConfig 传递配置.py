from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# 方式1：在调用时通过 config 传递
def process_with_api_key(input_data, config: RunnableConfig):
    """从 config 中获取 API key"""
    api_key = config.get("configurable", {}).get("api_key")
    if not api_key:
        raise ValueError("API key not provided")

    # 使用 API key
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    return llm.invoke(input_data["text"])


# 创建可运行对象
runnable = RunnableLambda(process_with_api_key)

# 调用时传递机密信息
config = RunnableConfig(
    configurable={
        "api_key": "sk-your-secret-key-here",
        "user_id": "user123",
        "session_token": "token-abc-123"
    }
)

result = runnable.invoke(
    {"text": "Hello, AI!"},
    config=config
)
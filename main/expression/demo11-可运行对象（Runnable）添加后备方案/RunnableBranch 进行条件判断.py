import os

from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# 定义不同的处理链
local_chain = (
    RunnablePassthrough()
    | ChatOllama(model="deepseek-r1:7b")
    | StrOutputParser()
)

cloud_chain = (
    RunnablePassthrough()
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

offline_chain = (
    RunnablePassthrough()
    | (lambda x: "抱歉，当前无法处理请求，请稍后再试。")
)

# 创建条件分支
def check_local_availability(input_data):
    """检查本地服务是否可用"""
    try:
        # 尝试连接本地 Ollama
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_cloud_availability(input_data):
    """检查云服务是否可用"""
    return bool(os.getenv("OPENAI_API_KEY"))  # 检查是否有 API Key

branch_chain = RunnableBranch(
    (check_local_availability, local_chain),   # 条件1：本地可用
    (check_cloud_availability, cloud_chain),   # 条件2：云服务可用
    offline_chain                               # 默认：离线模式
)

result = branch_chain.invoke("Hello")
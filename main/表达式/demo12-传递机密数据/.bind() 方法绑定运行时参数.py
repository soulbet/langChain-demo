from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 基础模型（不绑定 API key）
base_llm = ChatOpenAI(model="gpt-3.5-turbo")


# 在运行时动态绑定 API key
def create_chain_with_auth(api_key: str):
    """创建带有认证信息的链"""
    llm_with_auth = base_llm.bind(
        api_key=api_key,
        headers={"Authorization": f"Bearer {api_key}"}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("human", "{input}")
    ])

    return prompt | llm_with_auth


# 使用时
api_key = "sk-your-secret-key"
chain = create_chain_with_auth(api_key)
result = chain.invoke({"input": "Hello"})
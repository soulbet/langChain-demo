
from langchain.agents import create_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 1. 准备向量存储
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# 2. 描述向量存储的信息（工具包需要此信息来生成工具）
vectorstore_info = VectorStoreInfo(
    name="my_knowledge_base",
    description="包含公司内部政策文档的知识库",
    vectorstore=vectorstore
)

# 3. 实例化LLM和工具包
llm = ChatOpenAI(model="gpt-4o-mini")
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

# 4. 获取工具包中的所有工具
tools = toolkit.get_tools()

# 5. 创建智能体并使用
agent = create_agent(llm=llm, tools=tools)
response = agent.invoke({"messages": [("user", "公司今年的年假政策是什么？")]})
print(response)
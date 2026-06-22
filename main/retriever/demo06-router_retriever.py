"""
RouterRetriever 则更专注于动态选择。它使用一个 LLM 作为“路由器”，根据用户查询的内容，智能地决定将查询发送给哪一个特定的检索器。

适用场景：当你的系统拥有多个功能各异的数据源（如一个处理代码库、一个处理技术文档），需要根据问题类型自动分派查询时，RouterRetriever 非常合适。

"""
from langchain.chains.router import MultiRetrievalQAChain
from langchain.retrievers import RouterRetriever

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 注意：RouterRetriever的具体实现可能因版本有所不同，以下展示其核心思想

# 假设我们有两个不同的向量库
vectorstore_python = FAISS.from_documents(python_docs, OpenAIEmbeddings())
vectorstore_js = FAISS.from_documents(js_docs, OpenAIEmbeddings())

# 创建各自的检索器
retriever_python = vectorstore_python.as_retriever()
retriever_js = vectorstore_js.as_retriever()

# 使用 LLM 构建路由链
llm = ChatOpenAI(model="gpt-4o-mini")
router_chain = MultiRetrievalQAChain.from_retrievers(
    llm,
    retriever_infos=[
        {"name": "Python 文档", "description": "适用于Python编程问题", "retriever": retriever_python},
        {"name": "JavaScript 文档", "description": "适用于JavaScript编程问题", "retriever": retriever_js},
    ],
    verbose=True
)

# 自动路由查询
query = "如何用 Python 读取文件？"
response = router_chain.invoke(query)
print(response["result"])
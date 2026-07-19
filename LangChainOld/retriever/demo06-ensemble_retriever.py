"""
EnsembleRetriever 采用加权平均的方式，将多个不同检索器的结果进行整合与重排序。它的核心逻辑是：把多个检索器各自返回的文档按相关性排序，然后计算每个文档在不同结果列表中排名的倒数（即倒数排名融合, RRF），最后按加权得分重新排序。

适用场景：当你希望同时利用“关键词匹配”和“语义相似度”时，可以将 BM25Retriever（关键词检索）与 vectorstore.as_retriever()（语义检索）组合起来，实现优势互补。

"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. 准备文档
docs = [
    Document(page_content="我喜欢吃苹果，苹果是一种水果。"),
    Document(page_content="苹果公司发布了新款iPhone。"),
    Document(page_content="我喜欢骑自行车去公园。"),
]

# 2. 创建两种检索器
# BM25检索器（基于关键词）
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2  # 返回2个结果

# 向量检索器（基于语义）
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. 集成两个检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 权重相加等于1
)

# 4. 执行检索
query = "苹果"
results = ensemble_retriever.invoke(query)
for doc in results:
    print(doc.page_content)
    print("-" * 20)
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. 准备文档
docs = ["华为云ModelArts是面向AI开发者的平台。", "昇思MindSpore是一个全场景AI框架。", "ModelArts Pro是企业级AI应用开发套件。"]

# 2. 创建关键词检索器（BM25）
bm25_retriever = BM25Retriever.from_texts(docs)
bm25_retriever.k = 2

# 3. 创建向量检索器（语义搜索）
vector_store = FAISS.from_texts(docs, OpenAIEmbeddings())
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. 融合！创建混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 两种结果权重各占一半
)

# 5. 使用
query = "ModelArts平台是做什么的？"
retrieved_docs = ensemble_retriever.invoke(query)
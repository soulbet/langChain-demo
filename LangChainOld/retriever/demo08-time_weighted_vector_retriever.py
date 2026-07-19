from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import faiss

# 1. 初始化一个空的向量存储
embeddings_model = OpenAIEmbeddings()
embedding_size = 1536  # OpenAI 嵌入向量的维度

# 手动构建一个 FAISS 向量存储的底层实例
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# 2. 创建 TimeWeightedVectorStoreRetriever 实例
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,  # 衰减率，默认是 0.01
    k=1               # 返回的文档数量
)

# 3. 关键：必须通过检索器添加文档，而不是 vectorstore
retriever.add_documents([
    Document(page_content="我喜欢吃披萨。"),
    Document(page_content="意大利面是我最喜欢的食物。"),
    Document(page_content="我喜欢寿司。")
])

# 4. 执行检索
# 第一次查询，会返回最相关的文档（通常也是最新的）
results1 = retriever.invoke("我最喜欢的食物是什么？")
print(results1) # 输出最相关的文档

# 5. 再次查询同一个问题（此时第一次被访问的文档会更新“最后访问时间”）
results2 = retriever.invoke("我最喜欢的食物是什么？")
print(results2) # 输出可能会变化
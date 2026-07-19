from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

## 多向量检索（多模态数据）


# 假设我们有图片和对应的文本描述
image_docs = [
    Document(
        page_content="一张蓝色天空下盛开向日葵的图片",  # 图片描述（用于索引）
        metadata={"type": "image", "path": "/images/sunflower.jpg"}  # 实际存储路径
    ),
    Document(
        page_content="一张城市夜景中霓虹灯闪烁的图片",
        metadata={"type": "image", "path": "/images/city_night.jpg"}
    ),
]

# 创建向量存储和文档存储
vectorstore = FAISS.from_documents(
    image_docs,
    OpenAIEmbeddings()
)
docstore = InMemoryStore()

# 创建多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id",  # 文档ID的键名
    search_kwargs={"k": 2}
)

# 添加文档
for doc in image_docs:
    # 为每个文档生成唯一ID
    doc_id = f"img_{hash(doc.page_content)}"
    doc.metadata["doc_id"] = doc_id
    retriever.vectorstore.add_documents([doc])
    retriever.docstore.mset([(doc_id, doc)])

# 测试检索
query = "花朵的图片"
results = retriever.invoke(query)
print(f"查询: {query}")
for doc in results:
    print(f"检索到: {doc.page_content}")
    print(f"实际路径: {doc.metadata.get('path')}")
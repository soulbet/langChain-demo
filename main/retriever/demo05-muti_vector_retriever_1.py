"""
核心思想是，一个文档（或数据）可以被分解成多个部分，并为每个部分分别创建向量进行索引，但在检索时又能根据配置将这些部分组合或关联起来返回

它的主要应用场景包括：
多模态数据：比如一份包含文本和图像的PDF幻灯片。可以为图像总结（文本）创建向量索引，而实际存储和返回的却是原始图像本身。
父文档检索：为了提高检索精度，将长文档拆分成非常小的文本块进行索引（小块的向量更精确），但检索时返回其所属的更大的父文档块，为LLM提供更完整的上下文。
使用场景与示例
最适合数据源异构（非纯文本），或需要同时优化检索精度和上下文完整性的场景
"""
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. 准备长文档
parent_docs = [
    Document(
        page_content="""人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
        机器学习是AI的一个子集，它使系统能够从数据中学习并改进，而无需明确编程。
        深度学习是机器学习的一个子集，它使用神经网络来模拟人类大脑的工作方式。""",
        metadata={"doc_id": "ai_intro"}
    )
]

# 2. 定义分割器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)  # 父文档分割器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)   # 子文档分割器（更小）

# 3. 创建向量存储和文档存储
vectorstore = FAISS(OpenAIEmbeddings(), InMemoryStore())
docstore = InMemoryStore()

# 4. 创建父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2}  # 返回父文档数量
)

# 5. 添加文档
retriever.add_documents(parent_docs)

# 6. 测试检索
query = "什么是深度学习？"
results = retriever.invoke(query)
print(f"查询: {query}")
for doc in results:
    print(f"返回父文档: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
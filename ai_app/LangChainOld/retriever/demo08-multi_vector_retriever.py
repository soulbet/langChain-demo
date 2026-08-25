"""
LangChain 中的 MultiVectorRetriever 是实现此功能的基础组件。它通过分离向量存储（用于索引和搜索）和文档存储（用于存储原始文档），
允许你将一个文档的多个不同“表示”（如摘要、小文本块）向量化并用于检索，但最终返回完整的原始文档。

它最典型的两种应用模式是：
多模态检索：为图像、音频等非文本数据生成文本摘要，然后为这些摘要（而非文件本身）建立向量索引。当用户搜索时，系统会匹配到最相关的摘要向量，
然后从文档存储中返回对应的原始图片或文件。这种方式常用于构建幻灯片问答、图文混排文档的检索系统。
父文档检索：将一份长文档切分成多个小的文本块（“子块”）进行向量化索引，但将这些“子块”与它们所属的更大文本块（“父块”）或原始文档关联起来。
检索时，系统先找到最相关的“子块”，然后返回其对应的内容更丰富的“父文档”，为 LLM 提供更完整的上下文。

"""
import uuid
from typing import List

from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================
# 1. 准备原始文档（一份长文档）
# ============================================
original_doc = Document(
    page_content="""人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    机器学习是AI的一个子集，它使系统能够从数据中学习并改进，而无需明确编程。
    深度学习是机器学习的一个子集，它使用神经网络来模拟人类大脑的工作方式。
    自然语言处理是AI的一个领域，专注于使计算机能够理解、解释和生成人类语言。
    AI的应用包括图像识别、语音识别、推荐系统和自动驾驶等。""",
    metadata={"source": "AI简介", "author": "张教授"}
)

# ============================================
# 2. 为同一个文档生成多个"向量表示"（子块）
# ============================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,  # 较小的块，用于精确检索
    chunk_overlap=10,
    separators=["。", "，", "。", "；", " ", ""]
)

# 将原始文档拆分成多个子块
sub_docs = splitter.split_documents([original_doc])

print(f"原始文档被拆分成 {len(sub_docs)} 个子块")
for i, doc in enumerate(sub_docs):
    print(f"  子块 {i + 1}: {doc.page_content[:30]}...")

# ============================================
# 3. 创建多向量检索器
# ============================================
# 创建向量存储（用于索引子块的向量）
vectorstore = FAISS.from_documents(sub_docs, OpenAIEmbeddings())

# 创建文档存储（用于存储原始文档）
docstore = InMemoryStore()

# 创建多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id",  # 用于关联向量与原始文档的键名
    search_kwargs={"k": 3}  # 检索时返回3个文档
)

# ============================================
# 4. 将原始文档存入文档存储，并建立映射关系
# ============================================
# 为原始文档生成唯一ID
doc_id = str(uuid.uuid4())
original_doc.metadata["doc_id"] = doc_id

# 将原始文档存入文档存储（键为doc_id，值为原始文档）
retriever.docstore.mset([(doc_id, original_doc)])

# 关键步骤：将每个子块的向量与原始文档的ID关联起来
# 这样检索时找到子块的向量，就能映射回完整的原始文档
for sub_doc in sub_docs:
    # 在子块的元数据中记录原始文档的ID
    sub_doc.metadata["doc_id"] = doc_id

# 将这些带有映射关系的子块重新添加到向量存储
# 注意：这里重新添加会覆盖之前添加的子块，确保每个子块都带有正确的doc_id
retriever.vectorstore = FAISS.from_documents(sub_docs, OpenAIEmbeddings())

print("\n✅ 多向量索引构建完成！")


# ============================================
# 5. 执行检索测试
# ============================================
def test_retrieval(query: str):
    """测试检索功能"""
    print(f"\n{'=' * 50}")
    print(f"查询: {query}")
    print(f"{'=' * 50}")

    # 执行检索
    results = retriever.invoke(query)

    print(f"检索到 {len(results)} 个文档")
    for i, doc in enumerate(results, 1):
        print(f"\n📄 结果 {i}:")
        print(f"  内容: {doc.page_content[:100]}...")
        print(f"  元数据: {doc.metadata}")
        print(f"  {'-' * 40}")


# ============================================
# 6. 测试不同的查询
# ============================================
# 测试1: 关于深度学习的查询
test_retrieval("什么是深度学习？")

# 测试2: 关于自然语言处理的查询
test_retrieval("自然语言处理的应用")

# 测试3: 关于AI应用的查询
test_retrieval("AI在哪些领域有应用？")
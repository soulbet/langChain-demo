"""
核心是继承 BaseRetriever 类并实现其核心方法，从而将你的业务逻辑封装成一个标准的检索器接口。
这能让你灵活地接入任意数据源（如数据库、API、本地文件等），并能无缝集成到 LangChain 的 RAG 流程中

_get_relevant_documents	同步检索方法。必须实现，用于接收查询字符串，返回相关文档列表。
_aget_relevant_documents	异步检索方法。可选实现，用于支持异步操作，提升性能。
"""

from typing import List, Union
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



class TfidfRetriever(BaseRetriever):
    """
    一个基于 TF-IDF 关键词匹配的自定义检索器。
    它接收一组文档，并利用 TF-IDF 算法计算查询与文档的相关性。
    """
    documents: List[Document]  # 检索器管理的文档列表
    # 以下字段有默认值，无需在 __init__ 时传入
    corpus: List[str] = []
    vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix: any = None  # 使用 any 或更具体的类型

    def __init__(self, documents: List[Document]):
        super().__init__(documents=documents)
        self.documents = documents
        # 1. 准备文档文本
        self.corpus = [doc.page_content for doc in documents]
        # 2. 训练 TF-IDF 向量化器
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun
    ) -> list[Union[Document, list[Document]]]:
        """同步方法：根据查询返回相关文档"""
        # 1. 将查询文本转换为 TF-IDF 向量
        query_vec = self.vectorizer.transform([query])

        # 2. 计算查询与所有文档的余弦相似度
        similarities = self.tfidf_matrix.dot(query_vec.T).toarray().flatten()

        # 3. 按相似度降序排序，获取文档索引
        sorted_indices = np.argsort(similarities)[::-1]
        # 4. 构建带分数的文档列表
        docs_with_scores = []
        for i in sorted_indices:
            # 获取原始文档
            doc = self.documents[i]
            # 获取对应的相似度分数
            score = similarities[i]

            # 关键步骤：将分数添加到文档的元数据中
            # 注意：使用 doc.metadata.copy() 可以避免修改原始文档的元数据
            new_metadata = doc.metadata.copy()
            new_metadata["score"] = score
            # 创建新的 Document 对象，包含更新后的元数据
            new_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata
            )
            docs_with_scores.append(new_doc)

        # 4. 返回排序后的文档列表（这里返回所有文档，实际应用可限制数量）
        return [self.documents[i] for i in sorted_indices]

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """异步方法：根据查询返回相关文档"""
        # 直接复用同步逻辑，或实现真正的异步操作（如异步数据库查询）
        return self._get_relevant_documents(query, run_manager=run_manager)

# 1. 准备一些示例文档
docs = [
    Document(page_content="我喜欢吃苹果，苹果是一种水果。"),
    Document(page_content="我喜欢骑自行车去公园。"),
    Document(page_content="苹果公司发布了新款iPhone。"),
]

# 2. 创建自定义检索器实例
retriever = TfidfRetriever(documents=docs)

# 3. 执行检索
query = "苹果"

"""
在 LangChain 中，当一个可运行对象（Runnable）被调用（如 retriever.invoke(query)）时，其执行流程如下：

调用入口：首先会执行 invoke 方法，它负责处理一些公共的运行时逻辑（如管理回调、处理配置等）。

核心逻辑：invoke 内部会调用你实现的 _get_relevant_documents 方法，并返回结果。

异步支持：对于异步调用 ainvoke，它会调用你实现的 _aget_relevant_documents 方法。
"""
results = retriever.invoke(query)

# 4. 查看结果
for doc in results:
    print(doc.page_content)

# 输出：
# 苹果公司发布了新款iPhone。
# 我喜欢骑自行车去公园。
# 我喜欢吃苹果，苹果是一种水果。
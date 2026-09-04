from langchain_core.documents import Document
from sqlalchemy import text


class Tools:

    def __init__(
            self,
            vector_store,
            es_store,
            reranker,
            engine
    ):
        self.engine = engine
        self._vector_store = vector_store
        self._es_store = es_store
        self._reranker = reranker

    def rrf(
            self,
            vector_docs,
            keyword_docs,
            k=60
    ):

        scores = {}

        all_docs = {}

        for doc in vector_docs + keyword_docs:
            key = doc.page_content

            all_docs[key] = doc

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content

            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        for rank, doc in enumerate(keyword_docs):
            key = doc.page_content

            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            all_docs[item[0]]
            for item in ranked
        ]

    def retrieve_context(self, query: str) -> str:
        """
            根据用户问题进行混合检索。

            检索流程:
            1. PGVector 语义搜索
            2. PostgreSQL pg_trgm 关键词搜索
            3. RRF 融合排序
            4. BGE Reranker 重排序

            参数:
                query: 用户问题

            返回:
                最相关的文档上下文内容
            """

        if self._vector_store is None:
            return "错误：向量存储未初始化，请先加载文档。"

        # ==================================================
        # 1. Vector Search
        # ==================================================

        vector_results = (
            self._vector_store
            .similarity_search(
                query,
                k=20
            )
        )

        keyword_results = (
            self._es_store
            .search(
                query,
                k=20
            )
        )

        print(
            "\n========== Vector Search =========="
        )

        for i, d in enumerate(vector_results):
            print(
                i + 1,
                d.page_content[:50]
            )

        print(
            "\n========== Keyword Search =========="
        )

        for i, d in enumerate(keyword_results):
            print(
                i + 1,
                d.page_content[:50]
            )

        results = self.rrf(
            vector_results,
            keyword_results
        )

        print(
            "\n========== Hybrid / RRF =========="
        )

        for i, d in enumerate(results[:10]):
            print(
                i + 1,
                d.page_content[:50]
            )

        docs = self._reranker.rerank(
            query,
            results[:100],
            top_k=3
        )

        print("\n========== Reranker ==========")

        for i, doc in enumerate(docs):
            print(f"\n--- {i + 1} ---")
            print(doc.page_content[:200])

        # ==================================================
        # 5. Context
        # ==================================================

        context = "\n\n---\n\n".join(
            f"文档片段 {i + 1}：\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

        return context
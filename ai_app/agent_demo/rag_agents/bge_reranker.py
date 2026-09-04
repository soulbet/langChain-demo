from sentence_transformers import CrossEncoder


class BGEReranker:

    def __init__(
            self,
            model_name="BAAI/bge-reranker-v2-m3"
    ):
        print("加载 Reranker...")
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            device="cuda"
        )

    def rerank(
            self,
            query: str,
            docs: list,
            top_k: int = 3
    ):
        """
        docs: LangChain Document列表
        """

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, score in ranked[:top_k]
        ]
import asyncio
import os.path
import subprocess
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_postgres import PGVectorStore, PGEngine
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from sqlalchemy import create_engine, text, inspect
from tqdm import tqdm

from ai_app.model_factory.model_factory import ModelFactory
from utils.es_util import ElasticsearchBM25

# ==================== 全局变量 ====================
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


class RagAgent:

    def __init__(self, db, tbl_name, embedding, conn_str):
        self.db = db
        self.tbl_name = tbl_name
        self.embedding = embedding
        self.conn_str = conn_str

    def load_and_split_text(self, file_path: str):
        """加载文本并切分为片段"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {os.path.abspath(file_path)}")

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""],
            length_function=len,
        )
        chunks = text_splitter.split_documents(docs)
        print(f"✅ 文档已切分为 {len(chunks)} 个片段")
        return chunks

    def init_table(self, embedding_dim, engine):
        sql_engine = create_engine(
            self.conn_str
        )

        # ========================================
        # 3. 判断表是否存在
        # ========================================
        inspector = inspect(sql_engine)

        if not inspector.has_table(
                self.tbl_name,
                schema="public"
        ):
            print(f"表 '{self.tbl_name}' 不存在，创建向量表")

            engine.init_vectorstore_table(
                table_name=self.tbl_name,
                schema_name="public",
                vector_size=embedding_dim,
            )
            # ========================================
            # 5. 删除该文件以前的旧向量
            # ========================================
        else:
            print(f"表 '{self.tbl_name}' 已存在，直接使用")

    def delete_table(self, file_path):
        sql_engine = create_engine(
            self.conn_str
        )

        source = str(file_path)
        with sql_engine.begin() as conn:
            result = conn.execute(
                text(f"""
                                    DELETE FROM "public"."{self.tbl_name}"
                                    WHERE langchain_metadata->>'source' = :source
                                """),
                {
                    "source": source
                }
            )

            print(
                f"🗑️ 删除旧数据: {result.rowcount} 条"
            )

    # ==================== 存入 pgvector ====================
    def store_to_pgvector(self,
                          chunks,
                          file_path,
                          vector_store
                          ):
        """将文档片段存入 pgvector"""

        # ========================================
        # 4. 给每个 chunk 添加 source
        # ========================================
        source = str(file_path)

        for i, doc in enumerate(chunks):
            doc.metadata["source"] = source
            doc.metadata["chunk_id"] = i

        # ========================================
        # 7. 插入新数据
        # ========================================
        print("📥 写入文档...")

        texts = [
            doc.page_content
            for doc in chunks
        ]

        metadatas = [
            doc.metadata
            for doc in chunks
        ]

        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        es_store.add_documents(
            chunks
        )

        print(
            f"✅ 已存入 {len(chunks)} 个文档片段到 '{tbl_name}' 表"
        )


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


class DocTools:

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

    def keyword_search(
            self,
            query: str,
            k: int = 10
    ):
        """
        使用 PostgreSQL pg_trgm 进行关键词/文本检索
        """

        sql = text(f"""
            SELECT
                content,
                langchain_metadata,
                similarity(content, :query) AS score
            FROM "public"."{self.table_name}"
            WHERE similarity(content, :query) > 0.05
            ORDER BY score DESC
            LIMIT :k
        """)

        with self.sql_engine.begin() as conn:
            rows = conn.execute(
                sql,
                {
                    "query": query,
                    "k": k
                }
            ).fetchall()

        docs = []

        from langchain_core.documents import Document

        for row in rows:
            metadata = row.langchain_metadata or {}

            docs.append(
                Document(
                    page_content=row.content,
                    metadata=metadata
                )
            )

        return docs

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


class LlmAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def create_rag_agent(self, llm, tools):
        """使用 LangChain v1.x 的 create_agent 创建 RAG Agent"""
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=self.system_prompt,
        )

        return agent


# ==================== 主流程 ====================
if __name__ == "__main__":
    # ----- 7.1 准备 Embedding 模型 -----
    print("🚀 初始化 Embedding 模型...")
    embedding = ModelFactory().create_model(local_model_type='embed')
    db = 'agent_embedding'
    tbl_name = 'ai_agent_text'
    conn_str = (
        f"postgresql+psycopg://postgres:123456@localhost:5432/{db}"
    )
    engine = PGEngine.from_connection_string(
        url=conn_str
    )
    rag_agent = RagAgent(db=db,
                         tbl_name=tbl_name,
                         embedding=embedding,
                         conn_str=conn_str)
    embedding_dim = 1024
    rag_agent.init_table(embedding_dim=embedding_dim, engine=engine)

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=tbl_name,
        embedding_service=embedding,
    )
    ip_output = subprocess.check_output("wsl hostname -I", shell=True).decode()
    wsl_ip = ip_output.strip().split()[0]
    es_store = ElasticsearchBM25(
        hosts=f"http://{wsl_ip}:9200",
        index_name="ai_agent_text"
    )

    # ----- 7.2 加载文档 -----

    embed_flag = True

    if embed_flag:
        path = f"{Path(__file__).parent}/data/output"
        text_dir = os.listdir(path)
        for text_name in tqdm(text_dir):
            file_path = f"{path}/{text_name}"
            if Path(file_path).exists() and not Path(file_path).is_file():
                raise "文件不存在！"
            rag_agent.delete_table(file_path)

            print(f"📄 文件路径: {file_path}")
            chunks = rag_agent.load_and_split_text(str(file_path))

            # ----- 7.3 存入向量库 -----
            print("💾 存入向量库...")
            store = rag_agent.store_to_pgvector(
                chunks=chunks,
                file_path=file_path,
                vector_store=vector_store
            )

    # ----- 7.4 创建 Agent -----
    print("🤖 创建 Agent...")
    llm_agent = ModelFactory().create_model()
    reranker = BGEReranker()
    retriever = DocTools(
        vector_store,
        es_store,
        reranker,
        engine
    )

    retrieve_context_tool = tool(
        retriever.retrieve_context
    )
    tools = [retrieve_context_tool]
    system_prompt = """
        你是文档问答助手。

        回答问题前必须调用 retrieve_context。
        只能根据工具返回的文档内容回答，不得使用外部知识或编造。
        
        要求：
        1. 先理解文档，再用自己的话回答。
        2. 文档包含答案时，直接回答，不要说“未找到”。
        3. 文档只能回答部分内容时，只回答确定的部分。
        4. 文档确实没有答案时，回答“文档中未找到相关信息”。
        5. 回答简洁、准确。
    """

    agent = LlmAgent(system_prompt).create_rag_agent(llm_agent, tools)

    # ----- 7.5 测试查询 -----
    print("\n" + "=" * 50)
    # contexts = ["什么是微调？",
    #             "为什么需要对开源模型进行微调？",
    #             "文档中提到了哪些提高 AI 应用可靠性的方法？",
    #             "这本书中有没有介绍 Kubernetes 的部署方法？"]

    contexts = ["文档中有没有提到 Kubernetes？",
                "文档中有没有介绍 Kubernetes 的部署方法？",
                "文档中有没有介绍 Docker 的部署方法？",
                "文档中有没有介绍 GPU 模型部署？"]
    for context in contexts:
        print("=" * 10 + context + "=" * 10)
        response = agent.invoke({
            "messages": [{"role": "user", "content": context}]
        })

        final_message = response["messages"][-1]
        print("🤖 Agent回答:", final_message.content)

from langchain_postgres import PGEngine
from sqlalchemy import create_engine, inspect, text


class PgVectorStore:

    def __init__(self, ):
        pass

    def get_engine(self,conn_str):
        engine = PGEngine.from_connection_string(
            url=conn_str
        )
        return engine

    def init_table(self, embedding_dim, engine,conn_str,tbl_name):
        sql_engine = create_engine(
           conn_str
        )

        # ========================================
        # 3. 判断表是否存在
        # ========================================
        inspector = inspect(sql_engine)

        if not inspector.has_table(
                tbl_name,
                schema="public"
        ):
            print(f"表 '{tbl_name}' 不存在，创建向量表")

            engine.init_vectorstore_table(
                table_name=tbl_name,
                schema_name="public",
                vector_size=embedding_dim,
            )
            # ========================================
            # 5. 删除该文件以前的旧向量
            # ========================================
        else:
            print(f"表 '{tbl_name}' 已存在，直接使用")

    def delete_table(self, file_path,conn_str,tbl_name):
        sql_engine = create_engine(
              conn_str
        )

        source = str(file_path)
        with sql_engine.begin() as conn:
            result = conn.execute(
                text(f"""
                                    DELETE FROM "public"."{tbl_name}"
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
    def store_to_pgvector(
            self,
            chunks,
            file_path,
            vector_store,
            es_store,
            tbl_name
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

        es_store.add_documents(chunks)

        print(
            f"✅ 已存入 {len(chunks)} 个文档片段到 '{tbl_name}' 表"
        )
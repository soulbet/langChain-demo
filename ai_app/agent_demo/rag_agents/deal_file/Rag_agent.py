import asyncio
import os.path
import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_postgres import PGVectorStore

from tqdm import tqdm

from ai_app.agent_demo.rag_agents.bge_reranker import BGEReranker
from ai_app.agent_demo.rag_agents.llm_agent import LlmAgent
from ai_app.agent_demo.rag_agents.load_and_split_docs import LoadAndSplitDocs
from ai_app.agent_demo.rag_agents.pg_vector_store import PgVectorStore
from ai_app.agent_demo.rag_agents.tools import Tools
from ai_app.model_factory.model_factory import ModelFactory
from utils.es_util import ElasticsearchBM25
from utils.wsl_ip_util import WslIp

# ==================== 全局变量 ====================
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )



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

    pg_vector_store=PgVectorStore()
    engine = pg_vector_store.get_engine(conn_str=conn_str)

    embedding_dim = 1024
    pg_vector_store.init_table(embedding_dim=embedding_dim,
                         engine=engine,
                         conn_str=conn_str,
                         tbl_name=tbl_name
                         )

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=tbl_name,
        embedding_service=embedding,
    )
    wsl_ip = WslIp().get_wsl_ip()
    es_store = ElasticsearchBM25(
        hosts=f"http://{wsl_ip}:9200",
        index_name="ai_agent_text"
    )

    # ----- 7.2 加载文档 -----

    embed_flag = False

    if embed_flag:
        path = f"{Path(__file__).parent}/data/output"
        text_dir = os.listdir(path)
        for text_name in tqdm(text_dir):
            file_path = f"{path}/{text_name}"
            if Path(file_path).exists() and not Path(file_path).is_file():
                raise "文件不存在！"
            pg_vector_store.delete_table(file_path,conn_str,tbl_name)
            es_store.delete_by_source(file_path)

            print(f"📄 文件路径: {file_path}")
            chunks = LoadAndSplitDocs().load_and_split_text(str(file_path),
                                                            800,
                                                            120)

            # ----- 7.3 存入向量库 -----
            print("💾 存入向量库...")
            store = pg_vector_store.store_to_pgvector(
                chunks=chunks,
                file_path=file_path,
                vector_store=vector_store,
                es_store=es_store,
                tbl_name=tbl_name
            )

    # ----- 7.4 创建 Agent -----
    print("🤖 创建 Agent...")
    llm_agent = ModelFactory().create_model()
    reranker = BGEReranker()
    retriever = Tools(
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

    contexts = ["介绍下微调"]
    for context in contexts:
        print("=" * 10 + context + "=" * 10)
        response = agent.invoke({
            "messages": [{"role": "user", "content": context}]
        })

        final_message = response["messages"][-1]
        print("🤖 Agent回答:", final_message.content)

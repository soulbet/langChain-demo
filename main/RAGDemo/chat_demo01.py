import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_factory import model_factory


llm = model_factory().create_model()

# ==================== 第一步：加载文档 ====================

# 加载网页内容（这里换成中文内容更好的网页）
loader = WebBaseLoader(
    web_paths=("https://baike.baidu.com/item/人工智能",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("J-lemma-content",))
    ),
)
docs = loader.load()
print(docs[0].page_content)  # 查看抓取内容是否正确


# ==================== 第二步：分割文档 ====================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 每块1000字符
    chunk_overlap=200   # 重叠200字符，保证上下文连贯
)
splits = text_splitter.split_documents(docs)

# ==================== 第三步：向量化存储 ====================
# nomic-embed-text 英文模型   bge-m3 多语言模型
# ollama pull bge-m3
embedding = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# ==================== 第四步：创建历史感知检索器 ====================

# 根据聊天历史，将用户当前问题转化为独立问题
contextualize_q_system_prompt = (
    "根据聊天历史和用户的最新问题，"
    "可能需要引用聊天历史中的上下文信息，"
    "请将当前问题改写为一个独立、完整的问题。"
    "不要回答问题，只需在需要时改写问题，否则直接返回原问题。"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# ==================== 第五步：创建问答链 ====================

system_prompt = (
    "你是一个专业的问答助手。"
    "请根据以下检索到的上下文来回答用户的问题。"
    "如果根据上下文无法回答，请直接说'我不知道'。"
    "回答请控制在三句话以内，保持简洁。"
    "\n\n"
    "上下文：\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# ==================== 第六步：组合完整的RAG链 ====================

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ==================== 第七步：管理聊天历史 ====================

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取或创建聊天历史记录"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ==================== 第八步：测试 ====================

# 第一次提问
print("第一次提问：")
print(conversational_rag_chain.invoke(
    {"input": "什么是任务分解（Task Decomposition）？"},
    config={"configurable": {"session_id": "abc123"}}
)["answer"])

# 第二次提问（带有上下文引用）
print("\n第二次提问（依赖上下文）：")
print(conversational_rag_chain.invoke(
    {"input": "它有哪些常用的方法？"},
    config={"configurable": {"session_id": "abc123"}}
)["answer"])

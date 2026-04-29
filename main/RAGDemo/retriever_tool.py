import bs4
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from main.model_factory import model_factory

llm = model_factory().create_model()
### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embedding = OllamaEmbeddings(
    model="nomic-embed-text"  # 最强免费中文向量模型 把文字 → 数字向量
)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]
#
memory = MemorySaver()
config = {"configurable": {"thread_id": "abc123"}}  # 对！
agent_executor = create_react_agent(llm, tools,checkpointer=memory) # ReAct 智能体
# ========== 关键：一直聊天循环 ==========
print("===== AI知识库对话（输入 exit 退出）=====")
while True:
    user_input = input("你：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("对话结束")
        break
    # 调用智能体，带记忆上下文
    res = agent_executor.invoke({"input": user_input}, config)
    print("AI：", res["output"])
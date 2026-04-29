import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from main.model_factory import model_factory

# WebBaseLoader文档加载器集成
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 文档分割，每块1000个字符，块之间200个重叠，防止关键内容被切断
# 参数说明：
# - chunk_size: 每个文本块的最大字符数（1000个字符）
# - chunk_overlap: 相邻文本块之间的重叠字符数（200个字符），避免关键信息被切断
# - add_start_index: 在每个文本块中添加起始位置索引，便于追踪原文位置
# - length_function: 计算文本长度的函数，默认使用len()计算字符数
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200,
                                               add_start_index=True,
                                               length_function=len)
splits = text_splitter.split_documents(docs)

# 对我们的66个文本块进行索引，以便在运行时可以对它们进行搜索
#  将每个文档分割并将这些嵌入插入向量数据库 (或向量存储)

embedding = OllamaEmbeddings(
    model="nomic-embed-text"  # 最强免费中文向量模型 把文字 → 数字向量
)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

# 创建一个 “向量数据库检索器”  从你的知识库文档里，找出最相似的 6 条内容，丢给大模型去回答
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

# 从LangChain Hub拉取预定义的RAG提示模板
# 该模板定义了如何将检索到的上下文和用户问题组合成prompt，用于指导LLM生成回答

prompt = hub.pull("rlm/rag-prompt")

llm = model_factory().create_model()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 工作流程：
# 1. 接收用户问题，通过retriever从向量数据库检索相关文档
# 2. 使用format_docs将检索的文档格式化为字符串作为context
# 3. 将context和question传入prompt模板
# 4. 通过llm生成回答
# 5. 使用StrOutputParser解析输出结果为字符串
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)
# invoke()：普通调用 → 一次问一个问题，等全部回答完才返回
# stream()：流式返回 → 像 ChatGPT 一样一个字一个字吐出来
# abatch()：批量异步调用 → 一次问 N 个问题，并发跑，全部完了一起返回
for chunk in rag_chain.stream("What is Task Decomposition in RAG?"):
    print(chunk, end="", flush=True)

# cleanup
# vectorstore.delete_collection()

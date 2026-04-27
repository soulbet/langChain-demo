import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

# VectorStore转换为Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

print(len(retrieved_docs))
exit()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

# cleanup
vectorstore.delete_collection()

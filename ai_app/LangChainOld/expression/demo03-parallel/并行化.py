from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from main.RAGDemo.anay_video_txt_demo01 import llm

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
# 这里可以并行检索多个源
parallel_retriever = RunnableParallel(faiss_results=retriever)

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)


# RunnablePassthrough()：核心作用是将接收到的输入原封不动地传递给下一个组件，本身不做任何修改或处理
# RunnablePassthrough().assign()可以添加新字典
retrieval_chain = (
    {"context": parallel_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

retrieval_chain.invoke({"question": "where did harrison work", "language": "italian"})
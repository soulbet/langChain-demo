"""
可以检查chain内部步骤

"""
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from model_factory.model_factory import model_factory
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=embedding
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = model_factory().create_model()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
# 获得图形
chain.get_graph().print_ascii()
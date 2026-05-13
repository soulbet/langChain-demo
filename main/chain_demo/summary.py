from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

from main.model_factory import model_factory

"""
总结文本
Stuff，它简单地将文档连接成一个提示；
Map-reduce，适用于较大的文档集。这将文档分成批次，总结这些文档，然后总结这些摘要。
"""

llm = model_factory().create_model()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

# Instantiate chain
chain = create_stuff_documents_chain(llm, prompt)

# Invoke chain
result = chain.invoke({"context": docs})
print(result)
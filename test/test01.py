from main.model_factory import model_factory

llm = model_factory().create_model()

loader = UnstructuredEPubLoader("path/to/your/book.epub")
documents = loader.load()

response = llm.invoke("你能帮我整理电子书的知识点吗？")
print(response.content)
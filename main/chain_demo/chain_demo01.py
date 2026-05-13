from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from main.model_factory import model_factory

model = model_factory().create_model()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | model | StrOutputParser()


analysis_prompt = ChatPromptTemplate.from_template("is this a funny joke? {joke}")


# 字典会自动解析并转换为RunnableParallel，它并行运行所有值并返回一个包含结果的字典
composed_chain = {"joke": chain} | analysis_prompt | model | StrOutputParser()

print(composed_chain.invoke({"topic": "bears"}))
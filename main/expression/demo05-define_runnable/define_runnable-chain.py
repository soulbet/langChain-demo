from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

from model_factory import model_factory

prompt1 = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
prompt2 = ChatPromptTemplate.from_template("What is the subject of this joke: {joke}")

model = model_factory().create_model()
@chain
def custom_chain(text):
    prompt_val1 = prompt1.invoke({"topic": text})
    output1 = model().invoke(prompt_val1)
    parsed_output1 = StrOutputParser().invoke(output1)
    chain2 = prompt2 | model() | StrOutputParser()
    return chain2.invoke({"joke": parsed_output1})


custom_chain.invoke("bears")
"""

LangChain 的路由指的是根据输入动态选择不同的处理链（Chain），这个处理链可以是简单的提示词模板，也可以是一个完整的 Agent 或复杂的多步流程。
核心思路：先让一个“分类链”对用户的输入进行分类，然后在一个自定义函数中根据分类结果，返回对应的子链（而不是执行结果），
最后用 RunnableLambda 将这个函数包装起来

"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from ai_app.model_factory.model_factory import model_factory


# 1. 准备模型
model = model_factory().create_model()

# 2. 构建分类链：将用户问题分类为 "LangChain", "Anthropic" 或 "Other"
classification_chain = (
    PromptTemplate.from_template("""
    Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.
    Do not respond with more than one word.
    <question>{question}</question>
    Classification:""")
    | model
    | StrOutputParser()
)

# 3. 构建三个目标子链
langchain_chain = (
    PromptTemplate.from_template("""You are an expert in langchain. Always answer questions starting with "As Harrison Chase told me".
    Question: {question} Answer:""")
    | model
)

anthropic_chain = (
    PromptTemplate.from_template("""You are an expert in anthropic. Always answer questions starting with "As Dario Amodei told me".
    Question: {question} Answer:""")
    | model
)

general_chain = (
    PromptTemplate.from_template("Respond to: {question}") | model
)

# 4. 定义路由函数：根据分类结果返回对应的子链
def route(info):
    topic = info["topic"].lower()
    if "anthropic" in topic:
        return anthropic_chain
    elif "langchain" in topic:
        return langchain_chain
    else:
        return general_chain

# 5. 构建完整路由链
full_chain = (
    {
        "topic": classification_chain,           # 先执行分类链，得到分类结果
        "question": lambda x: x["question"]       # 同时保留原始问题
    }
    | RunnableLambda(route)                      # 路由函数决定下一步
)

# 6. 调用
response = full_chain.invoke({"question": "how do I use LangChain?"})
print(response.content)
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

from model_factory import model_factory

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


# 2. 路由函数：为每个新问题选择最匹配的模板
def prompt_router(input):
    # 2.1 将用户问题也转成向量
    query_embedding = embeddings.embed_query(input["query"])

    # 2.2 计算问题与各个模板的余弦相似度
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    # 得到: [0.23, 0.89]  (与physics相似度0.23，与math相似度0.89)

    # 2.3 选择相似度最高的模板
    most_similar = prompt_templates[similarity.argmax()]  # 选择math_template

    # 2.4 返回对应的PromptTemplate对象
    return PromptTemplate.from_template(most_similar)


chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)   # 动态选择模板
    | model_factory().create_model()
    | StrOutputParser()
)
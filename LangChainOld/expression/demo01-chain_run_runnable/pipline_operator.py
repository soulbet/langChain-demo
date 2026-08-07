from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import result

from model_factory.model_factory import model_factory

llm = model_factory().create_model()

prompt = ChatPromptTemplate.from_template("""
system: 你是一个 helpful 助手
user: 讲一个关于 {topic} 的笑话
""")
chain = prompt | llm | StrOutputParser()
chain.invoke({""})
# 在上面这个chain基础上，添加判断是否有趣的chain
analysis_prompt = ChatPromptTemplate.from_template(
    """
    这个笑话是否有趣？{joke}
    """
)
composed_chain = {"joke": chain} | analysis_prompt | llm | StrOutputParser()
# 或者
"""
composed_chain_with_pipe = (
    RunnableParallel({"joke": chain})
    .pipe(analysis_prompt)
    .pipe(model)
    .pipe(StrOutputParser())
)"""



answer_result = composed_chain.invoke({"topic": "熊"})
print(answer_result)

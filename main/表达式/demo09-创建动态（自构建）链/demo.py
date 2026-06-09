
"""
# 因为有chat_history，会触发contextualize_question链
# contextualize_question会将问题重写为：
# "What is the population of Egypt?"  （补充完整上下文）

# 最终检索得到：埃及人口约1.11亿


"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, chain

from main.model_factory import model_factory

llm= model_factory().create_model()

contextualize_instructions = """Convert the latest user question into a standalone question given the chat history. Don't answer the question, return the question and nothing else (no descriptive text)."""
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_instructions),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ]
)
contextualize_question = contextualize_prompt | llm | StrOutputParser()

qa_instructions = (
    """Answer the user question given the following context:\n\n{context}."""
)
qa_prompt = ChatPromptTemplate.from_messages(
    [("system", qa_instructions), ("human", "{question}")]
)


@chain
def contextualize_if_needed(input_: dict) -> Runnable:
    if input_.get("chat_history"):
        # NOTE: This is returning another Runnable, not an actual output.
        # 有历史 → 返回一个"问题重写链"（会结合上下文理解问题）
        return contextualize_question
    else:
        return RunnablePassthrough() | itemgetter("question")


@chain
def fake_retriever(input_: dict) -> str:
    return "egypt's population in 2024 is about 111 million"


full_chain = (
    RunnablePassthrough.assign(question=contextualize_if_needed).assign(
        context=fake_retriever
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

full_chain.invoke(
    {
        "question": "what about egypt",
        "chat_history": [
            ("human", "what's the population of indonesia"),
            ("ai", "about 276 million"),
        ],
    }
)

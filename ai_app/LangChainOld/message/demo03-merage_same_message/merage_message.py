from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

from ai_app.model_factory.model_factory import model_factory

llm = model_factory().create_model()


messages = [
    SystemMessage("你是一个乐于助人的助手。"),
    SystemMessage("你总是用笑话来回应。"),  # 与上一条 SystemMessage 连续
    HumanMessage([{"type": "text", "text": "为什么叫 LangChain？"}]),
    HumanMessage("Harrison 在追什么？"),   # 与上一条 HumanMessage 连续
    AIMessage('嗯，我想“WordRope”和“SentenceString”听起来没那么顺口吧！'),
    AIMessage("他可能是在追办公室里的最后一杯咖啡！"), # 与上一条 AIMessage 连续
]

# merge_message_runs 用于合并相同类型的连续消息，使对话历史更简洁、更符合模型处理习惯。
merged = merge_message_runs(messages)
chain = merged | llm
print("\n\n".join([repr(x) for x in merged]))
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from pydantic import BaseModel, Field

from model_factory.model_factory import model_factory


# 1. 定义一个简单的内存存储类，它继承自 BaseChatMessageHistory
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """一个简单的内存聊天历史存储实现。"""
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """添加新消息到历史中"""
        self.messages.extend(messages)

    def clear(self) -> None:
        """清空历史"""
        self.messages = []

# 这个字典用于在实际应用中存储不同session_id对应的历史对象
store = {}
llm = model_factory().create_model()
# 2. 实现关键的获取历史函数
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取对应的聊天历史对象。如果不存在，就创建一个新的。"""
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
# 1. 定义提示模板，其中 {history} 占位符将由 RunnableWithMessageHistory 自动填充
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位乐于助人的助手。请用中文回答问题。"),
    MessagesPlaceholder(variable_name="history"), # 历史消息将注入到这里
    ("human", "{input}")
])


# 3. 创建基础链
chain = prompt | llm

# 包装基础链，使其具备历史记忆能力
# 'input_messages_key' 指定输入字典中代表当前用户问题的键名
# 'history_messages_key' 指定提示模板中用于注入历史消息的键名
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_by_session_id,
    input_messages_key="input",  # 指定输入字典中的哪个key对应当前用户输入。
    history_messages_key="history", # 指定输入字典中的哪个key用于传递历史消息（与 MessagesPlaceholder 的变量名一致）。
)
# 4. 使用示例
session_id = "user_123_session_1"
# 第一轮对话
response1 = conversation.invoke(
    {"input": "你好，我叫小明。"},
    config={"configurable": {"session_id": session_id}}
)
print(f"AI: {response1.content}")

# 第二轮对话，模型会记得你的名字
response2 = conversation.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": session_id}}
)
print(f"AI: {response2.content}")
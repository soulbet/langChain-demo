"""

example_selector：负责为给定输入选择少量示例（以及返回的顺序）。这些实现了 BaseExampleSelector 接口。一个常见的例子是基于向量存储的 SemanticSimilarityExampleSelector
example_prompt：通过其 format_messages 方法将每个示例转换为 1 个或多个消息。一个常见的例子是将每个示例转换为一个人类消息和一个 AI 消息响应，或者一个人类消息后跟一个函数调用消息。
"""

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

from main.model_factory import model_factory

model = model_factory().create_model()
examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "2 🦜 4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]
# 创建了向量存储
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# 创建了向量存储后，我们可以创建 example_selector。在这里，我们将单独调用它，并将 k 设置为仅获取与输入最接近的两个示例
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
# 提示模板将通过调用 `select_examples` 方法来传递输入，从而加载示例。
example_selector.select_examples({"input": "horse"})

# 组装提示模板，使用上面创建的 example_selector
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

print(few_shot_prompt.invoke(input="What's 3 🦜 3?").to_messages())

# 将示例传递到另一个聊天提示模板
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

print(few_shot_prompt.invoke(input="What's 3 🦜 3?"))

chain = final_prompt | model

chain.invoke({"input": "What's 3 🦜 3?"})
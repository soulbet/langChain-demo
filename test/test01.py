from model_factory.model_factory import ModelFactory
from langchain_core.messages import SystemMessage, HumanMessage

llm = ModelFactory().create_model(local_model_type="coder")

# 1. 构建消息列表
messages = [
    SystemMessage(content="你是一个数学专家，擅长用 LaTeX 写出符合 Markdown 标准的公式。要求用 $$ ... $$ 包裹行间公式，用 \dfrac 代替 \frac。"),
    HumanMessage(content="请用 LaTeX 写出 Yeo-Johnson 变换的公式。")
]

# 2. 调用模型
response = llm.invoke(messages)

# 3. 打印结果
print(response.content)
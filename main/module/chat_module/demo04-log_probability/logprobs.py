"""
“日志概率”（Log Probability，简称 logprobs）是一个可以从 LLM（大语言模型）响应中获取的可选数据，
它代表了模型在生成每个 token（词/字）时，对该选择的“信心程度”或“确信度”

作用：
评估模型不确定性：通过观察日志概率的大小，你可以判断模型对自己的回答是否“自信”。如果概率都很低，可能意味着输入模糊或模型遇到了不熟悉的领域。

异常检测：在生成内容的质量控制中，可以设定一个阈值。当模型生成一个对数概率极低的 token 时（比如低于 -10），可能意味着它开始“胡说八道”了，从而触发警报或要求重新生成。

模型校准：在某些应用中，你可能希望模型的输出概率分布更符合真实世界的分布，日志概率数据是进行这种校准的基础。

精细控制生成过程：在一些高级生成策略中，你可以直接操作每一步 token 的对数概率，来调整最终的输出结果。

"""
from langchain_openai import ChatOpenAI

# 1. 初始化模型并绑定 logprobs 参数
DEEPSEEK_API_KEY = "your-deepseek-api-key" # 请替换为你的真实 API Key

llm = ChatOpenAI(
    model="deepseek-chat",                    # DeepSeek 模型名称
    openai_api_key="sk-83d73e8bca6948159a4f2d2b6d9abf94",          # 👈 DeepSeek 的 API Key
    openai_api_base="https://api.deepseek.com/v1", # 👈 DeepSeek 的 API 端点
    temperature=0.7,
    max_tokens=1024,
    logprobs=True,     # 开启 logprobs
    top_logprobs=5  # 5个备选
)

# 2. 调用模型
response = llm.invoke("空指针异常发生的原因")

# 3. 从 response_metadata 中提取日志概率数据
# 这里的 'content' 列表包含了每个输出 token 的日志概率信息
print(response)
print(response.response_metadata["logprobs"])


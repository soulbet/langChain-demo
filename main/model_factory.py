from langchain_openai import ChatOpenAI


class model_factory:
    def __init__(self):
        self.model = None

    def create_model(self):
        """
        kimi-k2:1t-cloud
        qwen3-coder:480b-cloud

        :return:
        """
        self.model = ChatOpenAI(
            model="deepseek-chat",  # DeepSeek 模型名称
            openai_api_key="sk-83d73e8bca6948159a4f2d2b6d9abf94",  # 👈 DeepSeek 的 API Key
            openai_api_base="https://api.deepseek.com/v1",  # 👈 DeepSeek 的 API 端点
            temperature=0.7,
            max_tokens=1024,
            logprobs=True,  # 开启 logprobs
            top_logprobs=5  # 5个备选
        )
        return self.model

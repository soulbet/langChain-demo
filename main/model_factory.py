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
            model="qwen3.7-plus",  # DeepSeek 模型名称
            openai_api_key="sk-06aa98d2df7d4b0e9b7b540df2b7c37a",  # 👈 DeepSeek 的 API Key
            openai_api_base="https://ws-m29q37nbk63foudn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",  # 👈 DeepSeek 的 API 端点
            temperature=0.7,
            max_tokens=1024,
            logprobs=True,  # 开启 logprobs
            top_logprobs=5  # 5个备选
        )
        return self.model

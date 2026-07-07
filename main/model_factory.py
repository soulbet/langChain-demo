from langchain_openai import ChatOpenAI


class model_factory:
    def __init__(self):
        self.model = None

    def create_model(self, model_type='local'):
        """
        kimi-k2:1t-cloud
        qwen3-coder:480b-cloud

        :return:
        """
        if model_type == 'local':
            # d9931c8d-76e5-11f1-9352-010101010000
            self.model = ChatOpenAI(
                model="qwen2.5-vl-7b",  # DeepSeek 模型名称
                openai_api_key="11",  # 👈 DeepSeek 的 API Key
                openai_api_base="http://localhost:11434/v1",
                # 👈 DeepSeek 的 API 端点
                temperature=0.7,
                max_tokens=1024,
                logprobs=True,  # 开启 logprobs
                top_logprobs=5  # 5个备选
            )
            return self.model
        elif model_type == 'qwen':
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

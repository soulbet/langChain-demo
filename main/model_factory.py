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
        self.model = ChatOpenAI(model="qwen3-coder:480b-cloud",  # 你的模型
                                openai_api_key="ollama",  # 👈 必须加这个！ollama 固定填 ollama
                                base_url="http://localhost:11434/v1",  # 👈 本地地址也要加temperature=0
                                temperature=0.7)
        return self.model

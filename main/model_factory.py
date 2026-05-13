from langchain_openai import ChatOpenAI


class model_factory:
    def __init__(self):
        self.model = None

    def create_model(self):
        self.model = ChatOpenAI(model="deepseek-v3.1:671b-cloud",  # 你的模型
                                openai_api_key="ollama",  # 👈 必须加这个！ollama 固定填 ollama
                                base_url="http://localhost:11434/v1",  # 👈 本地地址也要加temperature=0
                                temperature=0.7)
        return self.model

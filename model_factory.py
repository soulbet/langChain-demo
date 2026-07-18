import requests
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class model_factory:
    def __init__(self):
        self.model = None
        self.EMBEDDING_URL = "http://localhost:11435/v1/embeddings"

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
                model="qwen3.7-plus",  # 模型名称
                openai_api_key="sk-ws-H.EDPDIRX.RvWQ.MEUCIQCy_XNiddcuNxhNeTP2pAdlOrDpODM_u07DGIsi_j7JjQIgHW9Q2kpYOpJCvvkYVlpbx5adlqWkBUwZCJuJRyHOuIE",
                openai_api_base="https://ws-uaqt9rtbju81kl76.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",  # 👈 DeepSeek 的 API 端点
                temperature=0.7,
                max_tokens=1024,
                logprobs=True,  # 开启 logprobs
                top_logprobs=5  # 5个备选
            )
            return self.model

    def get_embedding(self, text: str) -> list:
        """ 嵌入式向量库 """
        response = requests.post(
            self.EMBEDDING_URL,
            json={"input": text},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(f"API 调用失败: {response.text}")

    def access_data_from_tavily(self):
        """
        获取外部实时数据
        TavilySearch（）参数说明：
            max_results	    返回结果数量	    5
            topic	        搜索主题：       general / news / finance	general
            search_depth	搜索深度：       basic / advanced	basic
            include_answer	是否包含        AI 生成的答案摘要	False
            time_range	    时间范围：       day / week / month / year	None
            include_domains	限定特定域名	    None
            exclude_domains	排除特定域名	    None

        :return:
        """
        tavily_search = TavilySearch(
            tavily_api_key="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",
            max_results=2,
            topic="general",
            search_depth="basic"
        )
        return tavily_search

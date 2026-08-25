import subprocess
from typing import Dict, Any, List

import requests
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from tavily import TavilyClient


class ModelFactory:
    def __init__(self):
        self.model = None
        self.EMBEDDING_URL = "http://localhost:11435/v1/embeddings"

    def create_model(self, model_type='local', local_model_type='agent'):
        """
        kimi-k2:1t-cloud
        qwen3-coder:480b-cloud
# d9931c8d-76e5-11f1-9352-010101010000
        :return:
        """
        if model_type == 'local':

            model_map = {
                "agent": "Qwopus3.5-4B-Coder-MTP", ## 启动命令 start-agent
                "coder": "qwen2.5-coder-7b",  ## start-coder
                "vl": "qwen2.5-vl-7b",   ## start-vl
                "embed": "nomic-embed-text" ## start-embed
            }
            port_map = {
                "agent": 55002,
                "coder": 55001,
                "vl": 55003,
                "embed": 55004
            }
            if local_model_type != 'embed':
                self.model = ChatOpenAI(
                    model=model_map[local_model_type],
                    openai_api_base=f"http://localhost:{port_map[local_model_type]}/v1",
                    openai_api_key="11",  # 👈 DeepSeek 的 API Key

                    # 👈 DeepSeek 的 API 端点
                    temperature=0.7,
                    # logprobs=True,  # 开启 logprobs
                    # top_logprobs=5  # 5个备选
                )
            else:
                # ============ Embedding (向量生成) ============
                wsl_ip = "127.0.0.1"
                try:
                    ip_output = subprocess.check_output("wsl hostname -I", shell=True).decode()
                    wsl_ip = ip_output.strip().split()[0]
                except:
                    pass
                self.model = LlamaCppEmbeddings(base_url=f"http://{wsl_ip}:55004")
            return self.model
        elif model_type == 'qwen':
            self.model = ChatOpenAI(
                model="qwen3.7-plus",  # 模型名称
                openai_api_key="sk-ws-H.EDPDIRX.RvWQ.MEUCIQCy_XNiddcuNxhNeTP2pAdlOrDpODM_u07DGIsi_j7JjQIgHW9Q2kpYOpJCvvkYVlpbx5adlqWkBUwZCJuJRyHOuIE",
                openai_api_base="https://ws-uaqt9rtbju81kl76.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                # 👈 DeepSeek 的 API 端点
                temperature=0.7,
                max_tokens=1024,
                logprobs=True,  # 开启 logprobs
                top_logprobs=5  # 5个备选
            )
            return self.model

class LlamaCppEmbeddings(Embeddings):

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []

        for i, text in enumerate(texts):
            print(f"\n========== Embedding {i + 1}/{len(texts)} ==========")
            print(f"文本长度: {len(text)}")
            print(f"文本前100字: {text[:100]}")

            try:
                embedding = self._embed(text)

                print(f"Embedding成功，维度: {len(embedding)}")

                results.append(embedding)

            except Exception as e:
                print(f"❌ 第 {i + 1} 个文本失败")
                print(f"文本长度: {len(text)}")
                print(f"文本内容: {text[:500]}")
                raise
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    async def aembed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(
        self,
        text: str
    ) -> List[float]:
        return self.embed_query(text)

    def _embed(self, text: str) -> List[float]:
        payload = {"input": text}

        try:
            response = requests.post(
                f"{self.base_url}/embedding",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            embedding = result[0]["embedding"]

            if isinstance(embedding[0], list):
                embedding = embedding[0]

            if isinstance(embedding[0], list):
                embedding = embedding[0]

            return embedding
            # ---------------------

        except Exception as e:
            print(f"Error embedding text: {e}")
            raise e

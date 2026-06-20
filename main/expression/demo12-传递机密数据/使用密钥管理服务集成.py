import boto3
from langchain_core.runnables import RunnableLambda
from typing import Optional
import json

from main.model_factory import model_factory


class AWSSecretsManagerRunnable:
    """集成 AWS Secrets Manager 的可运行对象"""

    def __init__(self, secret_name: str, region_name: str = "us-east-1"):
        self.secret_name = secret_name
        self.region_name = region_name
        self.secrets_client = boto3.client(
            service_name='secretsmanager',
            region_name=region_name
        )

    def _get_secret(self) -> dict:
        """从 AWS Secrets Manager 获取机密"""
        try:
            response = self.secrets_client.get_secret_value(
                SecretId=self.secret_name
            )
            secret = json.loads(response['SecretString'])
            return secret
        except Exception as e:
            raise RuntimeError(f"无法获取机密: {e}")

    def create_chain(self):
        """创建使用机密信息的链"""
        secret = self._get_secret()

        # 使用获取到的 API key
        llm = model_factory().create_model()

        return RunnableLambda(
            lambda x: llm.invoke(x['text']),
            name="secure_llm_chain"
        )


# 使用
secrets_manager = AWSSecretsManagerRunnable(
    secret_name="my-app/secrets",
    region_name="us-east-1"
)
secure_chain = secrets_manager.create_chain()
result = secure_chain.invoke({"text": "Hello"})
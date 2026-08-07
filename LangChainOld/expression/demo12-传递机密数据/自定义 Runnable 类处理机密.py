from langchain_core.runnables import Runnable, RunnableConfig
from typing import Optional, Dict, Any
import os
from getpass import getpass

from model_factory.model_factory import model_factory


class SecureRunnable(Runnable):
    """带机密管理的可运行对象"""

    def __init__(self, secret_provider=None):
        self.secret_provider = secret_provider or self._get_secrets_from_env

    def _get_secrets_from_env(self):
        """从环境变量获取机密"""
        return {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY")
        }

    def _prompt_for_secret(self, secret_name: str):
        """交互式获取机密"""
        return getpass(f"请输入 {secret_name}: ")

    def invoke(
            self,
            input: Dict[str, Any],
            config: Optional[RunnableConfig] = None,
            **kwargs
    ) -> Any:
        # 从多个来源获取机密
        secrets = {}

        # 1. 优先从 config 获取
        if config and config.get("configurable"):
            secrets.update(config["configurable"])

        # 2. 其次从环境变量
        env_secrets = self._get_secrets_from_env()
        secrets.update({k: v for k, v in env_secrets.items() if v})

        # 3. 最后交互式询问
        if not secrets.get("api_key"):
            secrets["api_key"] = self._prompt_for_secret("API Key")

        # 使用获取到的机密
        llm = model_factory().create_model()
        return llm.invoke(input["text"])


# 使用
secure_runnable = SecureRunnable()
result = secure_runnable.invoke({"text": "Hello"})
import os
import warnings

from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage)

os.environ["LANGCHAIN_SUPPRESS_WARNINGS"] = "true"
os.environ["LANGCHAIN_HIDE_DEPRECATION_WARNING"] = "true"
chat = ChatOpenAI(model="deepseek-v3.1:671b-cloud",  # 你的模型
                  openai_api_key="ollama",  # 👈 必须加这个！ollama 固定填 ollama
                  base_url="http://localhost:11434/v1",  # 👈 本地地址也要加temperature=0
                  )
response=chat.invoke([
    HumanMessage(
        content=(
            "你好")
    )
]
)
parser = StrOutputParser()
result = parser.invoke(response)
print()

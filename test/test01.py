from pathlib import Path
from langchain_openai import ChatOpenAI

from model_factory.model_factory import ModelFactory

# 1. 连接你的本地模型
llm = ModelFactory().create_model(local_model_type='coder')

# 2. 加载一个代理定义（假设你复制了 'agents/frontend-developer.md' 到当前目录）
agent_file = Path("frontend-developer.md")
if agent_file.exists():
    agent_prompt = agent_file.read_text(encoding="utf-8")
else:
    agent_prompt = "你是一个有用的编程助手。"

# 3. 调用模型
response = llm.invoke(agent_prompt + "\n\n用户问题：请帮我写一个 python中英文翻译")
print(response.content)
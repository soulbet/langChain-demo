import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # api_key=os.getenv("Qwen_API_KEY"),
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # model="qwen-plus",
    model="deepseek-v3.1:671b-cloud",
    messages=[
        ## 设定模型的行为和规则
        {"role": "system", "content": "我在学习大模型，需要你的帮助."},
        # 设定模型的回答
        {"role": "assistant", "content": "你是大模型专家，需要解答我的疑问，并用最详细的语言告诉我原理."},
        # 提问内容
        {"role": "user", "content": "大模型的岗位有哪些？"},
    ]
    ,
    stream=True
)
for chunk in completion:
    print(chunk.choices[0].delta.content,end="",flush=True)

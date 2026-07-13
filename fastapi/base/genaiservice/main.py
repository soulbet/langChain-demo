
from fastapi import FastAPI
from openai import OpenAI

app=FastAPI()
client = OpenAI(api_key="ollama",
    base_url="http://localhost:11434/v1",)


@app.get("/")
async def root_controller():
    return {"status": "healthy"}

@app.get("/chat")
async def chat_controller(prompt: str='Inspire me'):
    response = client.chat.completions.create(
        model="deepseek-v3.1:671b-cloud",
        messages=[
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": prompt},
        ],
    )
    statement=response.choices[0].message.content
    return {"statement": statement}
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

model = ChatOpenAI(model="deepseek-v3.1:671b-cloud",  # 你的模型
                  openai_api_key="ollama",  # 👈 必须加这个！ollama 固定填 ollama
                  base_url="http://localhost:11434/v1",  # 👈 本地地址也要加temperature=0
                  )
# 创建一个输出解析器
parser = StrOutputParser()

# 创建一个prompt
system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

chain = prompt_template | model | parser

result = chain.invoke({"language": "italian", "text": "hi"})

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
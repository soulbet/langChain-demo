from langchain.llms import OpenAI
import getpass
# 运行时输入OpenAI API Key
getpass = getpass.getpass()

openAI = OpenAI(openai_api_key=getpass)

llm = OpenAI()
print(llm.predict("你好"))

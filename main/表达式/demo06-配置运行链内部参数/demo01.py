from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model=ChatOpenAI(model="qwen3-coder:480b-cloud",  # 你的模型
                                openai_api_key="ollama",  # 👈 必须加这个！ollama 固定填 ollama
                                base_url="http://localhost:11434/v1",  # 👈 本地地址也要加temperature=0
                                temperature=ConfigurableField(
                                                    id="llm_temperature",
                                                    name="LLM Temperature",
                                                    description="The temperature of the LLM",
                                                )
                 )
model.with_config(configurable={"llm_temperature": 0.9}).invoke("pick a random number")

model.invoke("pick a random number")
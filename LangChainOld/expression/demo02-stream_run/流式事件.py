from langchain_core.output_parsers import JsonOutputParser

from model_factory.model_factory import model_factory

llm = model_factory().create_model()

chain = (llm | JsonOutputParser()).with_config({"tags": ["my_chain"]})
events = []
async def a():
    async for event in llm.astream_events("hello", version="v2",include_types=["chat_model"]):
        events.append(event)
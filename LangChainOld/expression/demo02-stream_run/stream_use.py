from model_factory.model_factory import model_factory

llm = model_factory().create_model()

chunks = []
for chunk in llm.stream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

# 异步
# 包进 async 函数
async def stream_test():
    chunks = []
    async for chunk in llm.astream("what color is the sky?"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)
# 运行异步函数
import asyncio
asyncio.run(stream_test())
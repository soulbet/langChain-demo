from model_factory.model_factory import model_factory

chat=model_factory().create_model()
for chunk in chat.stream("Write me a 1 verse song about goldfish on the moon"):
    print(chunk.content, end="|", flush=True)

async def astream_f():
    # 异步
    async for chunk in chat.astream("Write me a 1 verse song about goldfish on the moon"):
        print(chunk.content, end="|", flush=True)


# astream_events，主要用于捕获核心时间和过滤事件
"""
包含以下事件：
on_chat_model_start	聊天模型	模型开始处理，你可以在这里看到输入的提示词。
on_chat_model_stream	聊天模型	最关键的事件。模型生成了一个 token（或 token 块），你可以从这里拿到 AIMessageChunk。
on_chat_model_end	聊天模型	模型调用完成，你可以在这里获取完整的输出和 response_metadata（包含 token 用量）。
on_chain_start/stream/end	链（Chain）	代表链中某个步骤（如 RunnableSequence）的生命周期。
on_tool_start/end	工具（Tool）	Agent 调用工具时触发，可以获取工具的名称和传入的参数。

过滤事件：
当你的应用非常复杂时，事件的数量会很多。astream_events 提供了强大的过滤功能，让你可以只关注自己需要的事件
1、按组件名称过滤 (include_names)：只关心特定名称的组件 chain.astream_events("问题", include_names=["my_parser"]):
2、按组件类型过滤 (include_types)：只关心某个类型的事件，比如只看和聊天模型相关的。chain.astream_events("问题", include_types=["chat_model"]):
3、按标签过滤 (include_tags)：为你的组件打上标签，然后按标签过滤。  chain.astream_events("问题", include_tags=["my-chain"])
"""

async def astream_events_f():
    idx = 0
    async for event in chat.astream_events(
        "Write me a 1 verse song about goldfish on the moon", version="v1"
    ):
        idx += 1
        if idx >= 5:  # Truncate the output
            print("...Truncated")
            break
        print(event)
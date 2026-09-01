# 消息类型

- SystemMessage：用于引导 AI 行为，通常作为输入消息序列中的第一个传入。
- HumanMessage： 表示与聊天模型交互的人的消息。
- AIMessage： 表示来自聊天模型的消息。这可以是文本或请求调用工具。
- FunctionMessage / ToolMessage： 用于将工具调用结果传回模型的消息。
- AIMessageChunk / HumanMessageChunk / ...： 每种消息类型的块变体。
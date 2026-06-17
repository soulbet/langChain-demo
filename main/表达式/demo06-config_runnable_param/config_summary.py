
from langchain_core.runnables import RunnableConfigI, RunnableConfig

from main.model_factory import model_factory

model = model_factory().create_model()

# 配置链
# RunnableConfig 是一个包含多个字段的字典
# 调用 .invoke(), .stream() 或 .batch() 时，通过 config 参数传递
config = RunnableConfig(
    tags=["my-chain", "test"],  # 添加标签，便于追踪
    metadata={"user_id": "123"}, # 添加元数据，用于监控
    max_concurrency=5,          # 限制并发数
    recursion_limit=10,         # 限制嵌套调用深度
    timeout=30,                 # 设置超时时间（秒）
)

# 运行时配置
chain = some_chain.with_config({"tags": ["default-tag"]})
# 所有后续调用都会带上 "default-tag"

response = model.invoke("你好", config=config)


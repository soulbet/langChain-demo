from langchain.runnables.hub import HubRunnable

from langchain_core.runnables import ConfigurableField, RunnableConfig

# 创建可配置的 HubRunnable
base_prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    owner_repo_commit=ConfigurableField(
        id="hub_commit",
        name="Hub Commit",
        description="The Hub commit to pull from",
    )
)

# 版本1：使用默认版本
result_v1 = base_prompt.invoke({
    "question": "什么是LangChain？",
    "context": "LangChain是一个..."
})

# 版本2：动态切换到另一个版本
config = RunnableConfig(
    configurable={
        "hub_commit": "rlm/rag-prompt@commit_hash_abc123"
    }
)
result_v2 = base_prompt.invoke(
    {"question": "什么是LangChain？", "context": "..."},
    config=config
)
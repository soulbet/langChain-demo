from langchain.runnables.hub import HubRunnable

from langchain_core.runnables import ConfigurableField, RunnableConfig

"""
HubRunnable 是一个特殊的可运行对象，专门用来从 LangChain Hub 上拉取提示词模板，让提示词的管理和复用变得极其方便。

configurable_fields 则是一种声明式的配置方法，能让你将一个可运行对象的某个属性（比如 HubRunnable 的 owner_repo_commit，即具体的提示词版本）标记为“可在运行时动态指定”，从而在不修改代码的情况下，灵活切换链的内部行为
"""

# 创建可配置的 HubRunnable
base_prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    # 它是 HubRunnable 的一个属性，用来指定要拉取的具体版本
    owner_repo_commit=ConfigurableField(
        id="hub_commit", # 配置的唯一标识符，用于在调用时引用
        name="Hub Commit",  # 可读的名称
        description="The Hub commit to pull from",  # 说明这个配置是干什么的
    )
)

# 版本1：使用默认版本
result_v1 = base_prompt.invoke({
    "question": "什么是LangChain？",
    "context": "LangChain是一个..."
})

# 版本2：动态切换到另一个版本
# 通过 RunnableConfig 传入了一个覆盖配置
config = RunnableConfig(
    configurable={
        "hub_commit": "rlm/rag-prompt@commit_hash_abc123"
    }
)
result_v2 = base_prompt.invoke(
    {"question": "什么是LangChain？", "context": "..."},
    config=config
)
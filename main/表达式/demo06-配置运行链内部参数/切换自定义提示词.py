from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField

# 创建一个默认模板
prompt = PromptTemplate.from_template(
    "请写一篇关于{subject}的冷笑话"
).configurable_fields(
    template=ConfigurableField(          # 暴露 template 字段为可配置
        id="prompt_template",
        name="提示模板",
        description="运行时动态替换的模板内容"
    )
)

# 正常调用：使用默认模板
content = prompt.invoke({"subject": "程序员"})
print(content.to_string())  # 输出: "请写一篇关于程序员的冷笑话"

# 运行时切换为完全不同的模板
content = prompt.invoke(
    {"subject": "程序员"},
    config={"configurable": {
        "prompt_template": "请写一首关于{subject}的藏头诗"
    }}
)
print(content.to_string())  # 输出: "请写一首关于程序员的藏头诗"
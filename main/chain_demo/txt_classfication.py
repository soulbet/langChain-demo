from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from main.model_factory import model_factory

"""
将文本分类到标签
"""


# 提示词中文化，并强化约束
tagging_prompt = ChatPromptTemplate.from_template(
    """
请从以下文本中提取所需的信息。

你必须提取 'Classification' 函数中要求的所有属性。
严禁将任何字段留空或设为空值。如果你不确定，请做出最合理的猜测。

文本：
{input}
"""
)


# 模型字段描述中文化
class Classification(BaseModel):
    sentiment: str = Field(description="文本的情感倾向（如：积极、消极、中立）")

    aggressiveness: int = Field(
        description="文本的攻击性程度，评分范围从 1 到 10（1表示无攻击性，10表示极具攻击性）",
    )

    language: str = Field(description="文本所使用的语言（如：中文、英文、西班牙语）")


# LLM
llm = model_factory().create_model().with_structured_output(
    Classification,
    method="function_calling",
    strict=False  # 保持关闭，确保稳定性
)

tagging_chain = tagging_prompt | llm

# 测试文本也换成了中文
inp = "我非常高兴能认识你！我觉得我们会成为非常好的朋友！"
print(tagging_chain.invoke({"input": inp}))

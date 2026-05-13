from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from main.model_factory import model_factory

"""
构建一个提取链
"""


class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="人的名字")
    hair_color: Optional[str] = Field(
        default=None, description="如果知道人的头发颜色"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="以米为单位的身高")

class PersonList(BaseModel):
    persons: List[Person] = Field(default_factory=list, description="提取到的所有人物信息列表")

# 创建一个提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个基于中文的信息提取专家. "
            "只能从当前文本中提取信息，不能添加额外内容。"
            "提取规则："
            "1. 从文本中提取所有人物的信息。"
            "2. 每个人物包含name（名字）、hair_color（头发颜色）、height_in_meters（以米为单位的身高）。"
            "3. 身高需要将厘米转换为米（164cm → 1.64m），要加上单位cm或者m，返回字符串格式。"
            "4. 如果某个属性无法从文本中提取，返回None。",
        ),
        # 输入参数
        ("human", "{text}"),
    ]
)

llm = model_factory().create_model()
runnable = prompt | llm.with_structured_output(schema=PersonList, method="function_calling")

text = "徐婷是一个身高1.64m，黑色头发的女生.曦宝是她的女儿，身高80cm，头发颜色有点黄"
print(runnable.invoke({"text": text}))

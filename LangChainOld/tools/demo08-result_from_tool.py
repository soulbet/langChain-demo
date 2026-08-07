import random
from typing import List, Tuple
from langchain_core.tools import tool

"""
“工件”（Artifact）指的是工具执行后产生的、需要保留但通常不需要（或不适合）直接发给大语言模型（LLM）的完整、原始的数据对象
"""

# 在定义工具时，需要将 response_format 参数设置为 "content_and_artifact"。这样工具在执行后，需要返回一个包含 (content, artifact) 的元组
@tool(response_format="content_and_artifact")
def generate_random_ints(min: int, max: int, size: int) -> Tuple[str, List[int]]:
    """生成指定范围内的随机整数数组。"""
    array = [random.randint(min, max) for _ in range(size)]
    # 返回一个元组: (content, artifact)
    content = f"成功生成了 {size} 个在 [{min}, {max}] 范围内的随机整数。"
    return content, array  # array 将作为 artifact 被存储

# 要获取包含工件的完整 ToolMessage，你需要通过 ToolCall 的方式来调用工具，而不是直接调用
tool_call = {
    "name": "generate_random_ints",
    "args": {"min": 0, "max": 9, "size": 10},
    "id": "unique_call_id",  # 必须提供唯一ID
    "type": "tool_call",     # 必须指定类型
}
full_output = generate_random_ints.invoke(tool_call)

print(f"发给模型的内容: {full_output.content}")
# 输出: 发给模型的内容: 成功生成了 10 个在 [0, 9] 范围内的随机整数。

print(f"下游使用的工件: {full_output.artifact}")
# 输出: 下游使用的工件: [2, 8, 0, 6, ...]

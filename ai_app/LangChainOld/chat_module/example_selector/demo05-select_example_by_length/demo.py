from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Examples of a pretend task of creating antonyms.
# 创建反义词的模拟任务示例。
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    # The examples it has available to choose from.
    # 提供可选示例
    examples=examples,
    # The PromptTemplate being used to format the examples.
    # 提示词模板用于格式化示例
    example_prompt=example_prompt,
    # The maximum length that the formatted examples should be.
    # Length is measured by the get_text_length function below.
    # 格式化示例的最大长度
    # 长度由下方的 get_text_length_length 函数测量
    max_length=25,
    # The function used to get the length of a string, which is used
    # to determine which examples to include. It is commented out because
    # it is provided as a default value if none is specified.
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)

# FewShotPromptTemplate 是 LangChain 里的少样本提示词模板，用来拼接出完整给大模型的提示词
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    # 我们提供一个示例选择器，而不是示例
    example_selector=example_selector,
    example_prompt=example_prompt,
    # prefix（前缀）和suffix（后缀）就是完整提示词的头部固定内容和尾部固定内容，中间会插入example_selector筛选出来的少样本示例。
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
print(dynamic_prompt.format(adjective="big"))

# 在示例选择器里面增加示例
new_example = {"input": "big", "output": "small"}
dynamic_prompt.example_selector.add_example(new_example)
print(dynamic_prompt.format(adjective="enthusiastic"))
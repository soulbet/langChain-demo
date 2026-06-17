
"""
NGramOverlapExampleSelector 根据与输入的 ngram 重叠分数选择和排序示例。
ngram 重叠分数是一个介于 0.0 和 1.0 之间的浮点数（包括 0.0 和 1.0）。

N-Gram 是文本切分规则，把句子 / 词语按连续 N 个单元拆分：
    1-Gram（一元）：单个字 / 词
    例：happy → [happy]
    2-Gram（二元，Bigram）：连续 2 个单元
    例：very happy → [very happy]
    3-Gram（三元，Trigram）：连续 3 个单元
多用于计算文本相似度、匹配相似例句，LangChain 里常用它做样本检索。
"""

from langchain_community.example_selectors import NGramOverlapExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a fictional translation task.
examples = [
    {"input": "See Spot run.", "output": "Ver correr a Spot."},
    {"input": "My dog barks.", "output": "Mi perro ladra."},
    {"input": "Spot can run.", "output": "Spot puede correr."},
]

example_selector = NGramOverlapExampleSelector(
    # The examples it has available to choose from.
    # 可供选择的示例
    examples=examples,
    # The PromptTemplate being used to format the examples.
    # 用于格式化示例的模板
    example_prompt=example_prompt,
    # The threshold, at which selector stops.
    # It is set to -1.0 by default.
    # 相似度阈值
    # threshold = 0.0：只保留有重叠的样本
    # threshold = -1.0：关闭过滤，所有样本按「相似度从高到低」排序（最常用）
    # 数值越大，筛选越严格
    threshold=-1.0,
    # For negative threshold:
    # Selector sorts examples by ngram overlap score, and excludes none.
    # For threshold greater than 1.0:
    # Selector excludes all examples, and returns an empty list.
    # For threshold equal to 0.0:
    # Selector sorts examples by ngram overlap score,
    # and excludes those with no ngram overlap with input.
)
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the Spanish translation of every input",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)
print(dynamic_prompt.format(sentence="Spot can run fast."))

new_example = {"input": "Spot plays fetch.", "output": "Spot juega a buscar."}

example_selector.add_example(new_example)
print(dynamic_prompt.format(sentence="Spot can run fast."))

example_selector.threshold = 0.0
print(dynamic_prompt.format(sentence="Spot can run fast."))

example_selector.threshold = 0.09
print(dynamic_prompt.format(sentence="Spot can play fetch."))

example_selector.threshold = 1.0 + 1e-9
print(dynamic_prompt.format(sentence="Spot can play fetch."))
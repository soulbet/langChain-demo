
"""
NGramOverlapExampleSelector 根据与输入的 ngram 重叠分数选择和排序示例。ngram 重叠分数是一个介于 0.0 和 1.0 之间的浮点数（包括 0.0 和 1.0）。

选择器允许设置阈值分数。ngram 重叠分数小于或等于阈值的示例将被排除。默认情况下，阈值设置为 -1.0，因此不会排除任何示例，只会重新排序。将阈值设置为 0.0 将排除与输入没有 ngram 重叠的示例。
重叠分数 = (两个文本共有的 N-gram 数量) / (两个文本中 N-gram 的总数)
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
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The threshold, at which selector stops.
    # It is set to -1.0 by default.
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
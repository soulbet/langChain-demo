"""
不直接将示例输入到 FewShotPromptTemplate 对象中，而是将它们输入到一个名为 SemanticSimilarityExampleSelector 的 ExampleSelector 实现实例中。
该类根据输入与初始集中的少量示例的相似性选择少量示例。它使用嵌入模型计算输入与少量示例之间的相似性，并使用向量存储执行最近邻搜索

"""

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from main.prompt_template.使用少量示例.少量示例创建格式化器 import examples, example_prompt

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OllamaEmbeddings(model="bge-m3"),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1,
)

# Select the most similar example to the input.
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(
    prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)
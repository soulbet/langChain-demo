import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda

from main.model_factory import model_factory

model = model_factory().create_model()

def parse_or_fix(text: str, config: RunnableConfig):
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Fix the following text:\n\n\`\`\`text\n{input}\n\`\`\`\nError: {error}"
            " Don't narrate, just respond with the fixed data."
        )
        | model
        | StrOutputParser()
    )
    for _ in range(3):
        try:
            return json.loads(text)
        except Exception as e:
            # 这里也会将config参数传入，来执行get_openai_callback()
            text = fixing_chain.invoke({"input": text, "error": e}, config)
    return "Failed to parse"


from langchain_community.callbacks import get_openai_callback
# get_openai_callback()，是 LangChain 中一个非常实用的上下文管理器，它的核心作用是自动记录代码块中所有 OpenAI 调用的用量和花费。
with get_openai_callback() as cb:
    output = RunnableLambda(parse_or_fix).invoke(
        "{foo: bar}", {"tags": ["my-tag"], "callbacks": [cb]}
    )
    print(output)
    print(cb)
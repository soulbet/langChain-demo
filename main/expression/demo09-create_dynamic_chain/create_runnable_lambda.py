from langchain_core.runnables import RunnableLambda

from main.model_factory import model_factory


# 一个简单的自定义函数，根据输入长度决定如何处理
def dynamic_router(input: str):
    if len(input) > 10:
        return "这是一个长文本，需要摘要：\n"
    else:
        return "这是一个短查询，直接回答：\n"
model = model_factory().create_model()
# 将函数包装成 Runnable
# RunnableLambda 允许你将任何 Python 函数（同步或异步）包装成一个 LangChain 的 Runnable 对象，使其能无缝接入链中
router = RunnableLambda(dynamic_router)

# 在链中使用
chain = router | model
print(chain.invoke("LangChain 是一个用于开发由语言模型驱动的应用程序的框架。"))
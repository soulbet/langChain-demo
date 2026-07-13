
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from model_factory import model_factory

llm = model_factory().create_model()
# 1. 普通函数 → 可运行对象（为了能用管道符）
def log_result(x):
    print(f"结果是: {x}")
    return x

# 2. 回调传播（让自定义逻辑能在执行时被调用）
class MyCallback(BaseCallbackHandler):
    def on_chain_end(self, outputs, **kwargs):
        print(f"链执行完成")

# 组合使用
runnable_log = RunnableLambda(log_result)  # 转成可运行
chain = llm | runnable_log

# 传播回调
chain.invoke("hello", config={"callbacks": [MyCallback()]})
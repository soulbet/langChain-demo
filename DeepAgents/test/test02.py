from model_factory.model_factory import ModelFactory

llm = ModelFactory().create_model("qwen")
result= llm.invoke("请使用纯文本公式，写出 Yeo-Johnson 变换的公式。")
print(f"result:{result.content}")
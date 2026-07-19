from model_factory.model_factory import ModelFactory

llm = ModelFactory().create_model(local_model_type="agent")
result= llm.invoke("请使用纯文本公式，写出 Yeo-Johnson 变换的公式。")
print(result.content)
from ai_app.model_factory.model_factory import ModelFactory

llm = ModelFactory().create_model()

for i in range(20):
    resp = llm.invoke("你好")
    print(resp.content)
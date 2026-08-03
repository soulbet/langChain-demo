from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
print(len(model.encoder.layer))  # 输出层数
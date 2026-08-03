from transformers import pipeline

# 1. 加载预训练的情感分析 pipeline
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# 2. 直接使用
result = classifier("This movie is disgustingly good!")
print(result)  # 输出: [{'label': 'POSITIVE', 'score': 1.0}]
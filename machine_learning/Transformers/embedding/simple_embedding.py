import torch
import torch.nn as nn

from machine_learning.Transformers.nlp.BPE import SimpleBPETokenizer

# --- 1. 先运行你之前训练好的 BPE 分词器，得到真实词表和 ID ---
# 把 SimpleBPETokenizer 类的完整代码放在这里，然后用你的语料训练

corpus = [
    "我每天都要吃苹果",
    "苹果很好吃",
    "我喜欢吃红苹果",
    "香蕉也是水果",
    "苹果和香蕉我都爱吃",
    "吃香蕉",
    "苹果汁很好喝",
]

tokenizer = SimpleBPETokenizer(vocab_size=30)
tokenizer.train(corpus)

# --- 2. 用训练好的分词器，编码一句测试文本 ---
test_text = "我爱吃苹果"
input_ids = torch.tensor(tokenizer.encode(test_text))
print(f"分词结果: {tokenizer.tokenize(test_text)}")
print(f"ID序列: {input_ids}")

# --- 3. 定义嵌入层（词表大小用实际训练的）---
vocab_size = len(tokenizer.vocab)  # 这才是真实的词表大小
embed_dim = 6

embedding = nn.Embedding(vocab_size, embed_dim)

# --- 4. 查表 ---
word_vectors = embedding(input_ids)

print(f"输入 ID 形状: {input_ids.shape}")
print(f"输出向量形状: {word_vectors.shape}")
print(f"\n每个词的向量:")
for i, vec in enumerate(word_vectors):
    token = tokenizer.idx2word.get(input_ids[i].item(), "[UNK]")
    print(f"'{token}' (ID {input_ids[i].item():2d}) -> {vec.detach().numpy()}")
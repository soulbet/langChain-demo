import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

"""
词汇表构建：从语料中提取出 {'I', 'love', 'machine', 'learning', 'deep', 'enjoy'} 等词。

训练样本生成：为每个“中心词-上下文词”对生成训练数据，例如 ('love', 'I')、('love', 'machine') 等。

训练过程：模型通过不断调整 in_embed 和 out_embed 的权重，使得当输入“love”时，输出层能更准确地预测“I”和“machine”等上下文词。

词向量结果：最终 in_embed 中的每一行就代表了一个词的向量。语义相近的词（如 love 和 enjoy）的向量会逐渐靠近，表现为余弦相似度较高。

💡 关键点解析
nn.Embedding 就是你说的“查找表”：它接收一个索引（如 word_to_idx["love"]），返回对应的向量。这个矩阵的每一行就是你要的词向量。

前向传播是“猜词”：model(center_idx) 计算了中心词与所有词的相似度，得分最高的词就是模型“猜测”的上下文词。

反向传播是“纠正”：CrossEntropyLoss 将真实上下文词的索引作为目标，计算误差并反向传播，更新 in_embed 和 out_embed。正是这个更新过程，让词向量逐渐具备语义。
"""


# ============================================
# 1. 准备数据
# ============================================
# 虚拟语料库（句子列表）
corpus = [
    "I love machine learning".split(),
    "I love deep learning".split(),
    "I enjoy learning".split()
]

# 构建词汇表
word_set = set()
for sentence in corpus:
    word_set.update(sentence)
print(f"word_set:{word_set}")
vocab = list(word_set)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}, 词汇: {vocab}")

# 生成训练样本 (中心词 -> 上下文词)
window_size = 1  # 只考虑左右各1个词
training_pairs = []
for sentence in corpus:
    for i, center_word in enumerate(sentence):
        # 获取上下文词（窗口内的词，不包括中心词）
        context = []
        for j in range(-window_size, window_size + 1):
            if j == 0:
                continue
            if 0 <= i + j < len(sentence):
                context.append(sentence[i + j])
        # 为每个上下文词创建一个训练对
        for context_word in context:
            training_pairs.append((center_word, context_word))

print(f"生成的训练对数量: {len(training_pairs)}")
print("示例训练对:", training_pairs[:3])


# ============================================
# 2. 定义数据集
# ============================================
class SkipGramDataset(Dataset):
    def __init__(self, pairs, word_to_idx):
        self.pairs = pairs
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        center_idx = self.word_to_idx[center]
        context_idx = self.word_to_idx[context]
        # 返回的 label 是上下文词的索引，用于 CrossEntropyLoss
        return torch.tensor(center_idx, dtype=torch.long), torch.tensor(context_idx, dtype=torch.long)


dataset = SkipGramDataset(training_pairs, word_to_idx)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# ============================================
# 3. 定义 Skip-gram 模型
# ============================================
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # 输入层 → 隐藏层的权重矩阵 (这就是“查找表”)
        # 每一行是一个词的词向量
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        # 隐藏层 → 输出层的权重矩阵
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_word_idx):
        """
        前向传播
        center_word_idx: [batch_size] 中心词的索引
        """
        # 1. 从查找表中获取中心词的词向量
        center_emb = self.in_embed(center_word_idx)  # [batch_size, embedding_dim]

        # 2. 计算中心词与所有词的相似度（点积）
        # 将中心词向量与所有词向量进行点积，得到未归一化的分数
        # 等效于: center_emb @ self.out_embed.weight.T
        scores = torch.mm(center_emb, self.out_embed.weight.T)  # [batch_size, vocab_size]
        return scores


# ============================================
# 4. 训练模型
# ============================================
embedding_dim = 10  # 我们想得到的词向量维度
model = SkipGramModel(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()  # 多分类损失

print("\n开始训练...")
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for center_idx, context_idx in dataloader:
        optimizer.zero_grad()
        # 预测：给定中心词，预测上下文词的概率分布
        logits = model(center_idx)  # [batch_size, vocab_size]
        # 损失：与真实上下文词索引比较
        loss = criterion(logits, context_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# ============================================
# 5. 查看训练得到的词向量
# ============================================
print("\n训练完成！")
print("词的向量表示 (从 in_embed 中提取):")
embeddings = model.in_embed.weight.detach().numpy()
for word, idx in word_to_idx.items():
    print(f"{word}: {embeddings[idx][:5]}...")  # 只打印前5维

# ============================================
# 6. 验证语义相似性：计算“love”和“enjoy”的相似度
# ============================================
love_vec = embeddings[word_to_idx["love"]]
enjoy_vec = embeddings[word_to_idx["enjoy"]]
similarity = np.dot(love_vec, enjoy_vec) / (np.linalg.norm(love_vec) * np.linalg.norm(enjoy_vec))
print(f"\n'love' 和 'enjoy' 的余弦相似度: {similarity:.4f}")
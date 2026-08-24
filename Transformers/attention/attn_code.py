import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 三个投影矩阵,创建全连接层
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, X):
        """
        X: 输入向量矩阵，形状 [batch, seq_len, embed_dim]
        """
        # 1. 投影：生成 Q, K, V
        Q = self.W_Q(X)  # [batch, seq_len, embed_dim]
        K = self.W_K(X)
        V = self.W_V(X)

        # 2. 计算注意力分数矩阵：Q @ K^T
        #    最后两维是 [seq_len, embed_dim] @ [embed_dim, seq_len] = [seq_len, seq_len]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        # 3. 缩放
        d_k = self.embed_dim
        attn_scores = attn_scores / math.sqrt(d_k)

        # 4. Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 5. 加权聚合：attn_weights @ V
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


# --- 测试 ---
# 假设输入和之前一样，形状 [1, 4, 6]（batch=1, seq_len=4, embed_dim=6）
X = torch.randn(1, 4, 6)

attention_layer = SimpleSelfAttention(embed_dim=6)
output, attn_weights = attention_layer(X)

print("输入形状:", X.shape)  # [1, 4, 6]
print("输出形状:", output.shape)  # [1, 4, 6]
print("\n注意力权重矩阵 (第一个batch):")
print(attn_weights[0])  # [4, 4]
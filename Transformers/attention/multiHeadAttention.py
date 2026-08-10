import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度，如 64

        # Q、K、V 的联合投影矩阵（一次性投影，再拆分）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出投影矩阵
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        输入: [batch, seq_len, d_model]
        输出: [batch, n_heads, seq_len, d_k]
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)  # 拆分
        return x.transpose(1, 2)  # 把 n_heads 维度提到前面

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = Q.shape

        # 1. 线性投影
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. 拆分成多头
        Q = self.split_heads(Q)  # [batch, n_heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算注意力分数 Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, n_heads, seq_len, seq_len]

        # 4. 缩放
        scores = scores / math.sqrt(self.d_k)

        # 5. Mask（可选，GPT 的因果注意力用）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 6. Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. 加权 V
        attn_output = torch.matmul(attn_weights, V)  # [batch, n_heads, seq_len, d_k]

        # 8. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # 9. 最终投影
        output = self.W_O(attn_output)

        return output, attn_weights


# --- 测试 ---
batch_size = 2
seq_len = 4
d_model = 768
n_heads = 12

mha = MultiHeadAttention(d_model, n_heads)
x = torch.randn(batch_size, seq_len, d_model)

output, attn_weights = mha(x, x, x)

print(f"输入形状: {x.shape}")  # [2, 4, 768]
print(f"输出形状: {output.shape}")  # [2, 4, 768]
print(f"注意力权重形状: {attn_weights.shape}")  # [2, 12, 4, 4]
#  ↑ 12 个头，每个头有 4×4 的注意力矩阵
import torch
import torch.nn as nn


class SimpleLayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        # 对最后一维 (embed_dim) 求均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch, seq_len, 1]

        # 标准化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放 + 平移
        return self.gamma * x_norm + self.beta


# --- 测试 ---
# 模拟注意力层输出的向量（数值范围很大）
attn_output = torch.tensor([
    [[12.3, -8.7, 0.02, 15.6],
     [-0.003, 4.1, -5.2, 0.8]]
])
print(f"归一化前: {attn_output}")

ln = SimpleLayerNorm(embed_dim=4)
output = ln(attn_output)
print(f"\n归一化后: {output}")
print(f"\n归一化后均值: {output.mean(dim=-1)}")  # 应该接近 0
print(f"归一化后方差: {output.var(dim=-1)}")  # 应该接近 1
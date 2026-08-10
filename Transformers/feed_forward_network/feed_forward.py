import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, ffn_dim)  # 膨胀
        self.W2 = nn.Linear(ffn_dim, embed_dim)  # 压缩
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 现代 Transformer 标配

    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        # 1. 膨胀 + 激活
        x = self.activation(self.W1(x))  # [batch, seq_len, ffn_dim]

        # 2. Dropout,Dropout 是在训练时，随机把一部分神经元的输出“掐掉”（设为 0），强迫模型不依赖任何单一神经元，从而提升泛化能力。
        x = self.dropout(x)

        # 3. 压缩回原维度
        x = self.W2(x)  # [batch, seq_len, embed_dim]

        return x


# --- 测试 ---
ffn = FeedForward(embed_dim=768, ffn_dim=3072)
x = torch.randn(1, 4, 768)  # 模拟注意力层输出
output = ffn(x)
print(output.shape)  # [1, 4, 768]，和输入形状完全一样
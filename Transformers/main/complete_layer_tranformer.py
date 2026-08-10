import torch
import torch.nn as nn
import math


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        d_model: 嵌入维度（如 768）
        n_heads: 注意力头数（如 12）
        d_ff:    前馈网络中间维度（如 3072，通常是 d_model 的 4 倍）
        """
        super().__init__()

        # --- 子层 1：多头自注意力 ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # 使用 [batch, seq_len, d_model] 格式
        )

        # --- 子层 2：前馈网络 ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # --- 两个层归一化 ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Dropout ---
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        # --- 子层 1：自注意力 + 残差连接 ---
        # 前归一化 + 注意力
        attn_input = self.norm1(x)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = x + self.dropout(attn_output)  # 残差连接

        # --- 子层 2：前馈网络 + 残差连接 ---
        # 前归一化 + FFN
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output  # 残差连接

        return x


# --- 测试 ---
batch_size = 2
seq_len = 4
d_model = 768
n_heads = 12
d_ff = 3072

layer = TransformerEncoderLayer(d_model, n_heads, d_ff)

# 模拟输入（嵌入 + 位置编码后的向量）
x = torch.randn(batch_size, seq_len, d_model)

output = layer(x)

print(f"输入形状: {x.shape}")  # [2, 4, 768]
print(f"输出形状: {output.shape}")  # [2, 4, 768]
print(f"输入输出形状相同: {x.shape == output.shape}")  # True
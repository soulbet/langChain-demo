import torch
import torch.nn as nn
import math


# ============================================================
# 1. 激活函数 GELU
# ============================================================
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


# ============================================================
# 2. 因果注意力掩码，只能看自己和下一个，未来不能看
# ============================================================
def create_causal_mask(seq_len, device):
    """
    创建因果掩码矩阵 [seq_len, seq_len]
    下三角为 0（允许看），上三角为 -inf（禁止看未来）
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1
    )
    return mask


# ============================================================
# 3. 多头因果注意力
# ============================================================
class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q、K、V 投影 + 输出投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)  ## 在训练过程中，输入张量的一些元素随机归零，概率为：attr: ‘ p ’

    def split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]

    def forward(self, hidden_states, causal_mask=None):
        batch, seq_len, _ = hidden_states.shape

        Q = self.split_heads(self.W_Q(hidden_states))
        K = self.split_heads(self.W_K(hidden_states))
        V = self.split_heads(self.W_V(hidden_states))

        # 注意力分数 Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 加上因果掩码（禁止看未来）
        if causal_mask is not None:
            scores = scores + causal_mask

        # Softmax + 加权 V
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        return self.W_O(attn_output)


# ============================================================
# 4. 前馈网络
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model=768, d_ff=3072, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.gelu = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W2(self.dropout(self.gelu(self.W1(x))))


# ============================================================
# 5. GPT 解码器层
# ============================================================
class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        # 多头注意力
        self.self_attn = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        # 子层 1：因果自注意力 + 残差 + 层归一化
        attn_output = self.self_attn(x, causal_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 子层 2：前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


# ============================================================
# 6. GPT 模型
# ============================================================
class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12,
                 n_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # N 层解码器
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最终归一化 + 输出头
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)  # 输出词表大小的 logits

    def forward(self, input_ids):
        """
        input_ids: [batch, seq_len]
        返回: logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Token 嵌入 + 位置嵌入
        tok_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        x = self.dropout(tok_emb + pos_emb)

        # 因果掩码
        causal_mask = create_causal_mask(seq_len, device)

        # 通过 N 层解码器
        for layer in self.layers:
            x = layer(x, causal_mask)

        # 最终输出
        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        自回归生成文本
        input_ids: [1, prompt_len]
        """
        self.eval()
        device = input_ids.device
   
        for _ in range(max_new_tokens):
            # 截断超长序列
            input_ids_cond = input_ids[:, -self.max_len:]

            # 前向传播，代表了模型认为“这个词作为下一个词出现”的可能性分数（Logit，也叫置信度分数）
            logits = self(input_ids_cond)  # [1, seq_len, vocab_size]

            # 取最后一个位置的 logits，缩放温度
            # 控制模型输出文本的随机性（或说创造性）
            # 当T < 1（低温）时：分母变小，原本概率高的词会变得更高，概率低的词变得更低，整个分布变得非常尖锐。模型会几乎肯定地选择那个最高概率的词。
            #
            # 当T > 1（高温）时：分母变大，高分和低分词之间的概率差距被拉平了。这让那些“候选词”有了更多被选中的机会，输出的多样性因此大大增加。
            logits = logits[:, -1, :] / temperature

            # 采样下一个 token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # 拼接到输入序列
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 遇到结束符就停止（假设 0 是结束符）
            if next_token.item() == 0:
                break

        return input_ids


# ============================================================
# 7. 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("测试 GPT 模型结构")
    print("=" * 50)

    # 用小参数测试结构
    # 词表大小（词汇量）
    vocab_size = 1000
    # 模型维度（隐藏层大小/嵌入维度）
    d_model = 256
    # 多头注意力的头数
    n_heads = 8
    # 解码器层数（Transformer 块的数量）
    n_layers = 6
    # 前馈神经网络的中间层维度
    d_ff = 1024
    # 模型支持的最大序列长度
    max_len = 128

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len
    )

    # 随机输入
    batch_size = 2
    seq_len = 10
    # input_ids 的结构: 形状为 [batch_size, seq_len] 的二维整数张量 (2D Tensor)
    # input_ids 的意义: 表示输入到 GPT 模型的 token ID 序列集合。
    # - batch_size (批次大小): 并行处理的文本序列数量（此处为 2）。
    # - seq_len (序列长度): 每条文本序列包含的 token 数量（此处为 10）。
    # - 元素取值: 范围在 [0, vocab_size - 1] 之间的整数，对应词表中的具体 token 索引。
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    logits = model(input_ids)

    print(f"输入形状: {input_ids.shape}")  # [2, 10]
    print(f"输出 logits 形状: {logits.shape}")  # [2, 10, 1000]

    # 验证因果性：位置 i 的预测不应该受位置 i+1 的影响
    input_ids_test = input_ids.clone()
    input_ids_test[:, 5:] = 999  # 修改后 5 个位置
    logits_test = model(input_ids_test)
    # 前 4 个位置的预测应该和原始输入完全一样
    diff = (logits[:, :4, :] - logits_test[:, :4, :]).abs().max().item()
    print(f"因果性验证（前 4 位最大差异，应接近 0）: {diff:.6f}")

    # 测试生成
    print("\n" + "=" * 50)
    print("测试自回归生成")
    print("=" * 50)

    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"输入 token 数: {prompt.shape[1]}")
    print(f"生成后 token 数: {generated.shape[1]}")
    print(f"新生成 token 数: {generated.shape[1] - prompt.shape[1]}")
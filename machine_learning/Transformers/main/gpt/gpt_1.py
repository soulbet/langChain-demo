# -*- coding: utf-8 -*-
"""
    @Project: langChain-demo
    @File   : gpt_1.py
    @Author : zyf
    @Date   : 2026/8/31 20:47
    @Desc   :
    """
import torch
import torch.nn as nn
import math


# ============================================================
# 1. 激活函数 GELU（不变）
# ============================================================
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


# ============================================================
# 2. 因果掩码（训练时全序列用；Decode 时单 query 不再需要）
# ============================================================
def create_causal_mask(seq_len, device):
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1
    )
    return mask


# ============================================================
# ★ 新增 3a. RoPE：预计算旋转因子
# ============================================================
def precompute_freqs_cis(head_dim, max_len, theta=10000.0):
    """
    返回 [max_len, head_dim/2] 的复数旋转因子 e^{i·m·θ_j}
    head_dim 是每个头的维度（即 d_k）
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()               # 位置 0..max_len-1
    freqs = torch.outer(t, freqs)                   # [max_len, head_dim/2]
    return torch.polar(torch.ones_like(freqs), freqs)  # 复数，模长1


def apply_rope(x, freqs_cis):
    """
    x: [batch, n_heads, seq_len, head_dim]
    freqs_cis: [seq_len, head_dim/2] —— 对应本次输入的各位置旋转因子
    """
    # 两两一组视为复数 (实部, 虚部)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # 广播到 batch 和 heads 维度
    fc = freqs_cis.unsqueeze(0).unsqueeze(0)
    out = x_complex * fc                            # 复数乘法 = 旋转
    return torch.view_as_real(out).flatten(-2).to(x.dtype)


# ============================================================
# ★ 新增 3b. KV Cache
# ============================================================
class KVCache:
    """每层一个：预分配最大空间，顺序追加，只读前 pos 个"""
    def __init__(self, batch_size, max_len, n_heads, head_dim, dtype, device):
        self.k = torch.zeros(batch_size, n_heads, max_len, head_dim,
                             dtype=dtype, device=device)
        self.v = torch.zeros(batch_size, n_heads, max_len, head_dim,
                             dtype=dtype, device=device)
        self.pos = 0  # 已缓存的长度（也是下一个 token 的位置索引）

    def update(self, k_new, v_new):
        """
        k_new, v_new: [batch, n_heads, new_len, head_dim]
        返回到当前为止的全部 K、V（新 token 的位置 = self.pos）
        """
        L = k_new.shape[2]
        self.k[:, :, self.pos:self.pos + L] = k_new
        self.v[:, :, self.pos:self.pos + L] = v_new
        self.pos += L
        return self.k[:, :, :self.pos], self.v[:, :, :self.pos]


# ============================================================
# 4. 多头因果注意力（★ 加入 RoPE 和 KV Cache）
# ============================================================
class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, dropout=0.1, max_len=1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ★ RoPE 旋转因子：每个头的维度 d_k 独立计算，随模型一起移动
        # 以字符串名字注册 buffer
        self.register_buffer(
            "freqs_cis",  # ① 名字，字符串
            precompute_freqs_cis(self.d_k, max_len), # ② 张量本身
            persistent=False # ③ 是否存入 state_dict
        )

    def split_heads(self, x):
        """
        将输入张量的特征维度拆分为多头格式，以便进行多头并行注意力计算。
        
        参数:
            x: 输入张量，形状为 [batch, seq_len, d_model]
            
        返回:
            拆分后的张量，形状为 [batch, n_heads, seq_len, d_k]
        """
        batch, seq_len, _ = x.shape
        # 将最后一维 d_model 拆分为 n_heads 和 d_k (d_model = n_heads * d_k)
        # 形状变化: [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k]
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        # 交换 seq_len 和 n_heads 维度，使多头维度提前，便于后续注意力矩阵乘法
        # 形状变化: [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def forward(self, hidden_states, causal_mask=None, cache=None):
        """
        
        # 1. 线性投影：将输入 hidden_states 分别通过权重矩阵映射到 Query、Key、Value 空间
        # 2. 多头拆分：调用 split_heads 将特征维度拆分为多头 (n_heads, d_k)，
        #    并调整张量形状为 [batch, n_heads, seq_len, d_k]，以支持多头并行注意力计算
        Q = self.split_heads(self.W_Q(hidden_states))  # 生成 Query，后续将应用 RoPE 旋转以注入位置信息
        K = self.split_heads(self.W_K(hidden_states))  # 生成 Key，后续将应用 RoPE 旋转以注入位置信息
        V = self.split_heads(self.W_V(hidden_states))  # ★ 生成 Value，V 不参与 RoPE 旋转，直接保留原始语义特征用于加权求和
        :return: 
        """
        batch, seq_len, _ = hidden_states.shape

        Q = self.split_heads(self.W_Q(hidden_states))
        K = self.split_heads(self.W_K(hidden_states))
        V = self.split_heads(self.W_V(hidden_states))   # ★ V 不旋转

        # ★ 根据 cache 确定本次的位置范围
        #   训练/无cache: 位置 0..seq_len-1
        #   Decode:       位置 cache.pos..cache.pos+seq_len-1
        if cache is not None:
            start = cache.pos
        else:
            start = 0
        fc = self.freqs_cis[start:start + seq_len].to(hidden_states.device)

        Q = apply_rope(Q, fc)   # ★ 只旋转 Q 和 K
        K = apply_rope(K, fc)

        # ★ 有 cache：追加新 K/V 并取回全部历史（K/V 长度 > 当前 seq_len）
        #   无 cache：直接用本序列的 K/V
        if cache is not None:
            K_full, V_full = cache.update(K, V)
        else:
            K_full, V_full = K, V

        scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ★ 因果掩码只在没有 cache 时需要（全序列训练路径）
        #   Decode 时 query 长度 1，缓存全是过去 token，天然因果
        if causal_mask is not None:
            scores = scores + causal_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V_full)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model)
        return self.W_O(attn_output)


# ============================================================
# 5. 前馈网络（不变）
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
# 6. GPT 解码器层（★ 透传 cache）
# ============================================================
class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1, max_len=1024):
        super().__init__()
        self.self_attn = MultiHeadCausalAttention(d_model, n_heads, dropout, max_len)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, cache=None):
        attn_output = self.self_attn(x, causal_mask, cache)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x


# ============================================================
# 7. GPT 模型（★ 删除绝对位置编码，加 cache 支持）
# ============================================================
class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12,
                 n_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_len = max_len

        # ★ 删除了 self.position_embedding —— RoPE 取代绝对位置编码
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, d_ff, dropout, max_len)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, caches=None):
        """
        input_ids: [batch, seq_len]
        caches:    None（训练，全序列）或 每层一个 KVCache 的 list（推理）
        返回: logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # ★ 只剩 token 嵌入，位置信息由 RoPE 在每层注意力内部注入
        x = self.dropout(self.token_embedding(input_ids))

        # ★ 因果掩码：只在全序列路径（无 cache）时构建
        causal_mask = create_causal_mask(seq_len, device) if caches is None else None

        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, causal_mask, cache)

        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        ★ 两阶段生成：Prefill 一次算完 prompt → Decode 每步只算 1 个 token
        """
        self.eval()
        batch = input_ids.shape[0]
        device = input_ids.device

        # ---- 为每一层建缓存 ----
        caches = [KVCache(batch, self.max_len, self.n_heads, self.d_k,
                          self.token_embedding.weight.dtype, device)
                  for _ in range(len(self.layers))]

        # ---- Prefill：整个 prompt 一次性前向，填满缓存 ----
        logits = self(input_ids, caches)          # [batch, prompt_len, vocab]

        # ---- Decode：每步只喂 1 个 token ----
        generated = []
        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature
            probs = torch.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            generated.append(next_token)

            if next_token.item() == 0:
                break

            # 只前向 1 个 token！历史 K/V 从缓存读取
            logits = self(next_token, caches)     # [batch, 1, vocab]

        return torch.cat(generated, dim=-1)


# ============================================================
# 8. 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("测试 GPT 模型结构（RoPE + KV Cache）")
    print("=" * 50)

    vocab_size, d_model, n_heads, n_layers, d_ff, max_len = 1000, 256, 8, 6, 1024, 128

    model = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, max_len=max_len)

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出 logits 形状: {logits.shape}")

    # ---- 因果性验证（训练路径，掩码仍然生效）----
    input_ids_test = input_ids.clone()
    input_ids_test[:, 5:] = 999
    logits_test = model(input_ids_test)
    diff = (logits[:, :4, :] - logits_test[:, :4, :]).abs().max().item()
    print(f"因果性验证（前 4 位最大差异，应接近 0）: {diff:.6f}")

    # ---- ★ KV Cache 正确性验证：缓存路径 vs 全量重算，logits 必须一致 ----
    print("\n" + "=" * 50)
    print("验证 KV Cache 数值等价性")
    print("=" * 50)

    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 8))

    # 路径 A：Prefill + 逐步 Decode（走缓存）
    caches = [KVCache(1, max_len, n_heads, d_model // n_heads,
                      torch.float32, prompt.device) for _ in range(n_layers)]
    with torch.no_grad():
        logits_a = model(prompt, caches)              # prefill
        step1_a = logits_a[:, -1, :]                  # prompt 最后位置的预测

        # 手动推进两步
        caches2 = [KVCache(1, max_len, n_heads, d_model // n_heads,
                           torch.float32, prompt.device) for _ in range(n_layers)]
        logits_b = model(prompt, caches2)
        next_tok = logits_b[:, -1, :].argmax(-1, keepdim=True)
        logits_b = model(next_tok, caches2)           # decode 1 步

    # 路径 B：不用 cache，把 prompt+新token 整段全量前向
    full_seq = torch.cat([prompt, next_tok], dim=-1)
    with torch.no_grad():
        logits_full = model(full_seq)

    diff_step1 = (step1_a - logits_full[:, 7, :]).abs().max().item()
    diff_step2 = (logits_b[:, -1, :] - logits_full[:, 8, :]).abs().max().item()
    print(f"Prefill 最后位置差异: {diff_step1:.6f}（应≈0）")
    print(f"Decode 1 步后差异:    {diff_step2:.6f}（应≈0，浮点误差内）")

    # ---- 生成测试 ----
    print("\n" + "=" * 50)
    print("测试自回归生成（Prefill + KV Cache）")
    print("=" * 50)
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"输入 token 数: {prompt.shape[1]}")
    print(f"生成 token 数: {generated.shape[0] if generated.dim()==1 else generated.shape[1]}")

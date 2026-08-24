import torch
import torch.nn as nn
import math


# ============================================================
# 1. 激活函数 GELU（你已学过）
# ============================================================
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


# ============================================================
# 2. 多头自注意力（你已学过）
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 64

        # Q、K、V 联合投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """

        将输入向量“切分”成多个独立的“头”，并为后续并行计算注意力做好准备。
        :param x:
        :return:
        """
        batch, seq_len, _ = x.shape
        # 重塑维度 # 维度变化: [2, 4, 512] → [2, 4, 8, 64]
        # 含义：把512维的向量，拆成8组，每组64维
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # [batch, n_heads, seq_len, d_k]

    def forward(self, hidden_states, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape

        Q = self.split_heads(self.W_Q(hidden_states))
        K = self.split_heads(self.W_K(hidden_states))
        V = self.split_heads(self.W_V(hidden_states))

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 让模型在计算注意力时，“忽略”掉那些不应该看到的位置（比如填充符 Padding 或未来的词）
        # 加上注意力掩码（padding 位置置为极小值）
        if attention_mask is not None:
            # attention_mask: [batch, 1, 1, seq_len] 或 [batch, seq_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            scores = scores + attention_mask

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权 V
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # 最终投影
        return self.W_O(attn_output)


# ============================================================
# 3. 前馈网络（你已学过）
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
# 4. Transformer 编码器层（你已学过）
# ============================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        # 子层 1：注意力 + 残差（后归一化，BERT 原始做法）
        attn_output = self.self_attn(hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = self.norm1(hidden_states)

        # 子层 2：FFN + 残差
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        hidden_states = self.norm2(hidden_states)

        return hidden_states


# ============================================================
# 5. BERT 嵌入层（Token + 位置 + 段嵌入）
# ============================================================
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, d_model=768, max_len=512, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)  # 词嵌入
        self.position_embeddings = nn.Embedding(max_len, d_model)  # 位置嵌入
        self.token_type_embeddings = nn.Embedding(2, d_model)  # 段嵌入（句子A/B）
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        tok_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + tok_emb  # 三种嵌入相加
        return self.dropout(self.norm(embeddings))


# ============================================================
# 6. 完整 BERT 模型
# ============================================================
class BERT(nn.Module):
    def __init__(self, vocab_size=30522, d_model=768, n_heads=12,
                 n_layers=12, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, dropout)

        # 12 层 Transformer 编码器
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        input_ids: [batch, seq_len]
        token_type_ids: [batch, seq_len]（可选）
        attention_mask: [batch, seq_len]（1 表示有效，0 表示 padding）
        """
        # 处理注意力掩码
        if attention_mask is not None:
            # 把 [batch, seq_len] 变成 [batch, 1, 1, seq_len]
            # 0 的位置设为极小值，1 的位置设为 0
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        else:
            extended_attention_mask = None

        # 嵌入
        hidden_states = self.embeddings(input_ids, token_type_ids)

        # 通过 12 层编码器
        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)

        # pooler_output: 取 [CLS] 向量（第一个 token），经过 tanh
        pooler_output = self.pooler(hidden_states[:, 0, :])

        # 返回两种输出
        return {
            "last_hidden_state": hidden_states,  # [batch, seq_len, 768]
            "pooler_output": pooler_output  # [batch, 768]
        }


# ============================================================
# 7. BERT + 分类头（用于情感分析等任务）
# ============================================================
class BERTForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = BERT()
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # 用 [CLS] 的向量做分类
        logits = self.classifier(outputs["pooler_output"])
        return logits


# ============================================================
# 8. 测试：用真实的中文预训练权重
# ============================================================
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # --- 先测试我们自己写的 BERT 结构是否正确 ---
    print("=" * 50)
    print("测试自定义 BERT 结构")
    print("=" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT().to(device)
    input_ids = torch.randint(0, 30522, (2, 10)).to(device)  # batch=2, seq_len=10
    attention_mask = torch.ones(2, 10).to(device)  # 全为有效 token
    attention_mask[0, 8:] = 0  # 第 0 句后两个是 padding

    outputs = model(input_ids, attention_mask=attention_mask)
    print(f"last_hidden_state 形状: {outputs['last_hidden_state'].shape}")  # [2, 10, 768]
    print(f"pooler_output 形状: {outputs['pooler_output'].shape}")  # [2, 768]

    # --- 测试分类头 ---
    print("\n" + "=" * 50)
    print("测试分类模型")
    print("=" * 50)

    cls_model = BERTForSequenceClassification(num_labels=2).to(device)
    logits = cls_model(input_ids, attention_mask=attention_mask)
    print(f"分类 logits 形状: {logits.shape}")  # [2, 2]
    print(f"预测类别: {torch.argmax(logits, dim=-1)}")

    # --- 加载真实的 HuggingFace BERT 中文权重 ---
    print("\n" + "=" * 50)
    print("加载真实 BERT 中文权重")
    print("=" * 50)

    try:
        from transformers import BertModel, BertForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        real_bert = BertModel.from_pretrained("bert-base-chinese")

        text = "今天天气真好，我们出去玩吧。"
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            real_outputs = real_bert(**inputs)

        print(f"输入文本: {text}")
        print(f"Token 数量: {inputs['input_ids'].shape[1]}")
        print(f"last_hidden_state 形状: {real_outputs.last_hidden_state.shape}")
        print(f"pooler_output 形状: {real_outputs.pooler_output.shape}")
        print(f"\n[CLS] 向量（前 10 维）: {real_outputs.pooler_output[0, :10]}")

    except ImportError:
        print("未安装 transformers，跳过真实模型加载")
        print("安装命令: pip install transformers")
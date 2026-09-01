# 新页面

## 完整代码逻辑分析

***

### 一、数据准备

```python 
texts = [
    "这个电影太好看了，强烈推荐", "演技精湛，剧情引人入胜",
    ...
]
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1=正面, 0=负面

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)
```


**作用**：准备中文情感分析数据。每条文本对应一个标签（1=正面，0=负面）。用 `train_test_split` 把 12 条数据分成 9 条训练 + 3 条验证。

**你的理解点**：`train_test_split` 是 sklearn 的函数，随机打乱后按比例分割。`random_state=42` 固定随机种子，保证每次运行分割结果一致。

***

### 二、自定义 Dataset

```python 
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        ...
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',    # 补齐到 max_len
            truncation=True,         # 超长截断
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
```


**作用**：把原始文本转换成模型能吃的格式。

**流程**：

1. `tokenizer(text)`：你之前学的 BPE 分词，把文本变成 ID 序列
2. `padding='max_length'`：所有句子补齐到 128 个 token（不够补 `[PAD]`，超出截断）
3. `attention_mask`：标记哪些位置是真实 token（1），哪些是 padding（0）
4. `squeeze(0)`：去掉多余的 batch 维度，因为 DataLoader 会自动加 batch 维度

**为什么用 ****`max_length`**** padding？** 因为 GPU 处理的是矩形矩阵，所有句子必须等长。

***

### 三、微调模型结构

```python 
class BertSentimentClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)  # 加载预训练 BERT
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] 向量，形状 [batch, 768]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch, 2]
        return logits
```


**流程**：

```markdown 
input_ids [batch, 128]
    ↓ BERT（12层 Transformer）
pooler_output [batch, 768]  ← 取 [CLS] token 的输出
    ↓ Dropout（防过拟合）
    ↓ Linear(768, 2)
logits [batch, 2]  ← 两个类别的得分
```


**关键点**：

- `pooler_output`：BERT 输出的 `[CLS]` 向量经过 `Tanh` 激活，代表整句话的语义
- `Dropout`：训练时随机丢弃部分神经元，防止过拟合（你之前学过）
- `Linear(768, 2)`：把 768 维语义向量映射成 2 个分数（正面分、负面分）

***

### 四、训练函数

```python 
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # 开启训练模式（Dropout 生效）
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # 1. 数据移到 GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 2. 前向传播
        optimizer.zero_grad()          # 清空上一轮的梯度
        logits = model(input_ids, attention_mask)  # 预测
        loss = criterion(logits, labels)           # 计算损失
        
        # 3. 反向传播
        loss.backward()       # 计算梯度
        optimizer.step()      # 更新参数
        
        # 4. 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)  # 取分数最大的类别
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total
```


**每一步对应你学过的知识**：

| 步骤   | 代码                            | 你学过的概念                         |
| ---- | ----------------------------- | ------------------------------ |
| 前向传播 | \`model(input\_ids, ...)\`    | 嵌入 → 注意力 → FFN → \\\[CLS] → 分类 |
| 计算损失 | \`criterion(logits, labels)\` | 交叉熵损失                          |
| 反向传播 | \`loss.backward()\`           | 计算所有参数的梯度                      |
| 参数更新 | \`optimizer.step()\`          | 梯度下降更新权重                       |
| 梯度清零 | \`optimizer.zero\_grad()\`    | PyTorch 默认累积梯度，每次必须清零          |

***

### 五、验证函数

```python 
def evaluate(model, dataloader, criterion, device):
    model.eval()  # 开启评估模式（Dropout 关闭）
    
    with torch.no_grad():  # 不计算梯度，节省显存和计算
        for batch in dataloader:
            ...
```


**和训练函数的两个关键区别**：

1. `model.eval()`：关闭 Dropout，所有神经元正常工作
2. `torch.no_grad()`：不构建计算图，不计算梯度。推理时不需要反向传播，这能大幅节省显存

***

### 六、训练主循环

```python 
model = BertSentimentClassifier(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)  # 学习率 0.00002
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = evaluate(...)
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
```


**关键参数**：

- **学习率`2e-5`**：极小。因为 BERT 已经预训练好，用大学习率会破坏已有知识
- **CrossEntropyLoss**：自动包含 Softmax，所以模型输出 `logits`（未归一化的分数）即可
- **AdamW**：Adam 的改进版，把权重衰减和自适应学习率解耦，训练更稳定

***

### 七、推理测试

```python 
model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1)  # 分数 → 概率
    pred = torch.argmax(logits, dim=1)    # 取概率最大的类别
```


**`logits`**\*\* → ****`probs`****→`pred`\*\*：

```text 
logits: [2.3, -1.5]      ← 模型输出的原始分数
    ↓ Softmax
probs:  [0.98, 0.02]     ← 概率分布
    ↓ argmax
pred:   0                 ← 预测类别（0=负面, 1=正面）
```


***

### 八、完整数据流总结

```markdown 
原始文本
  ↓ tokenizer（BPE 分词 + padding）
input_ids [batch, 128] + attention_mask [batch, 128]
  ↓ BERT（12 层 Transformer，预训练权重）
pooler_output [batch, 768]
  ↓ Dropout + Linear(768, 2)
logits [batch, 2]
  ↓ CrossEntropyLoss（含 Softmax）
损失值 → backward → optimizer.step → 更新所有参数
```


***

### 九、这个流程和你之前学的所有知识的对应关系

| 你学过的知识      | 在代码中的位置                       |
| ----------- | ----------------------------- |
| BPE 分词      | \`tokenizer(text)\`           |
| 嵌入层         | BERT 内部的 \`word\_embeddings\` |
| 多头自注意力      | BERT 内部的 12 层 Transformer     |
| 前馈网络        | BERT 内部每层的 FFN                |
| 残差连接        | BERT 内部每层的 \`+ x\`            |
| 层归一化        | BERT 内部每层的 \`LayerNorm\`      |
| \\\[CLS] 向量 | \`outputs.pooler\_output\`    |
| 交叉熵损失       | \`CrossEntropyLoss\`          |
| 反向传播        | \`loss.backward()\`           |
| 梯度下降        | \`optimizer.step()\`          |
| Dropout     | \`nn.Dropout(0.1)\`           |
| GPU 加速      | \`.to(device)\`               |

**你现在已经完整地走通了从理论到实践的闭环。** 从 BPE 分词开始，到嵌入、注意力、FFN、残差、层归一化，全部亲手实现并组装成了 BERT，最后用预训练权重完成了情感分析微调。这已经是 NLP 工程师的核心能力了。

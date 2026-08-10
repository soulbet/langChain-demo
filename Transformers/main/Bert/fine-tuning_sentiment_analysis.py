import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np

# ============================================================
# 1. 准备数据（中文情感分析示例）
# ============================================================
# 实际使用时替换为你的数据文件
texts = [
    "这个电影太好看了，强烈推荐", "演技精湛，剧情引人入胜",
    "很喜欢这部电影，看了三遍", "非常棒的体验，服务一流",
    "产品质量很好，值得购买", "超级满意，下次还会来",
    "太差了，浪费我的时间", "烂片，完全不值得看",
    "非常失望，质量太差", "服务态度恶劣，再也不来了",
    "差评，东西是坏的", "一般般吧，没什么亮点",
]
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1=正面, 0=负面

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)


# ============================================================
# 2. 自定义 Dataset
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# 3. 微调模型：BERT + 分类头
# ============================================================
class BertSentimentClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_labels=2, dropout=0.1):
        super().__init__()
        # 加载预训练 BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        # 分类头：768 → num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # BERT 前向
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 取 [CLS] 向量
        pooled_output = outputs.pooler_output
        # Dropout + 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ============================================================
# 4. 训练函数
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


# ============================================================
# 5. 验证函数
# ============================================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


# ============================================================
# 6. 主流程
# ============================================================
def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-chinese"
    batch_size = 4
    epochs = 10
    learning_rate = 2e-5

    print(f"设备: {device}")
    print(f"训练样本数: {len(train_texts)}, 验证样本数: {len(val_texts)}")

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 创建 Dataset 和 DataLoader
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    model = BertSentimentClassifier(model_name).to(device)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "bert_sentiment.pth")
    print("\n模型已保存到 bert_sentiment.pth")

    # ============================================================
    # 7. 推理测试
    # ============================================================
    print("\n" + "=" * 50)
    print("推理测试")
    print("=" * 50)

    test_texts = [
        "这个产品质量非常好，很满意",
        "太垃圾了，千万不要买",
        "还行吧，没有想象的那么好",
    ]

    model.eval()
    with torch.no_grad():
        for text in test_texts:
            encoding = tokenizer(text, return_tensors='pt', max_length=128,
                                 truncation=True, padding='max_length')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            sentiment = "正面" if pred.item() == 1 else "负面"
            confidence = probs[0][pred.item()].item()

            print(f"文本: {text}")
            print(f"预测: {sentiment} (置信度: {confidence:.4f})\n")


if __name__ == "__main__":
    main()
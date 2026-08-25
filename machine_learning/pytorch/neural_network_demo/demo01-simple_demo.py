import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # flatten会保留第0维，从第一维展平到最后一维
        # 全连接层需要二维输入，
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失（分类问题）
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 随机梯度下降

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # 数据集总大小（60000）
    model.train() # 控制的是模型内部层的行为 设置为训练模式（启用dropout等），随机丢弃神经元 使用当前batch的均值/方差
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # 移到GPU（如果可用）

        # 前向传播
        pred = model(X)             # 预测
        loss = loss_fn(pred, y)     # 计算损失

        # Backpropagation
        loss.backward()             # 计算梯度并累加到param.grad
        optimizer.step()            # 更新参数,更新的是新的权重
        optimizer.zero_grad()       # 清空梯度，删除梯度

        # 每100个batch打印一次损失
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)      # 测试集大小（10000）
    num_batches = len(dataloader)       # batch数量
    model.eval()                        # 控制的是模型内部层的行为 设置为评估模式 全部保留神经元   使用训练时累积的均值/方差
    test_loss, correct = 0, 0
    with torch.no_grad():               # 禁用梯度计算（节省内存）
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches            # 平均损失
    correct /= size                     # 准确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)     # 训练一个epoch
    test(test_dataloader, model, loss_fn)                  # 测试一次
print("Done!")
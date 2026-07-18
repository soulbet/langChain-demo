import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

"""
神经网络中的许多层都是参数化的，即具有与之关联的权重和偏置，在训练过程中会进行优化。
通过继承 nn.Module，可以自动跟踪模型对象内部定义的所有字段，并通过模型的 parameters() 或 named_parameters() 方法访问所有参数。
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


model = NeuralNetwork().to(device)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)

"""
dim=0（第 0 维）：代表批次大小。顺着 dim=0 往下看，你会看到不同的样本。
dim=1（第 1 维）：代表类别数量。顺着 dim=1 往右看，你会看到同一个样本在不同类别上的得分。
"""
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
print(model)
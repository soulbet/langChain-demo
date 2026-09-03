import torch.nn as nn

a = nn.Linear(10, 10)
print("权重:\n", a.weight)
print("偏置:\n", a.bias)
b=a(3)
print("权重:\n", b.weight)
print("偏置:\n", b.bias)
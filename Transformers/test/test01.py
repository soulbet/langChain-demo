import torch
from torch import nn

a=torch.randint(1,10,(3,4))
b=torch.triu(a,diagonal=1)*float('-inf')
print(b)

print(nn.Embedding(10, 7))

print(torch.arange(3, ).unsqueeze(0))
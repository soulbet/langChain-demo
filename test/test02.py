from torch import nn

a=nn.Linear(6, 6, bias=False)
print(a.weight)
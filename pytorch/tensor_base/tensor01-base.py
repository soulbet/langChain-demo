import torch
import numpy as np

""" 生成tensor的方式 """

# 直接从数据生成tensor
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# numpy的array生成
np_array = np.array(data)
x_np = torch.tensor(np_array)

# 保留了x_data的特性
x_ones = torch.ones_like(x_data)

# 也可以重写
x_rand = torch.rand_like(x_data, dtype=torch.float)

shape=[2,3]
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

""" tensor的属性 """

print(ones_tensor.shape)
print(ones_tensor.dtype)
print(ones_tensor.device)

""" tensor的计算 """
tensor=torch.ones([4,4])

print(f"first row:{tensor[0]}")
print(f"first column:{tensor[:,0]}")
print(f"last row:{tensor[...,-1]}")
tensor[:,1]=0
print(f"new tensor:\n{tensor}")

# cat() 拼接不增加维度
t1 = torch.cat([tensor,tensor,tensor,tensor],dim=1)
print(t1)

# 堆叠 增加维度
x_stack = torch.stack([tensor,tensor,tensor],dim=0)
print(x_stack)
print(x_stack.shape)
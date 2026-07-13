import numpy as np

## 用 block 将四个 2x2 数组连接成一个 4x4 数组
A = np.ones((2, 2))
B = np.eye(2, 2)
C = np.zeros((2, 2))
D = np.diag((-3, -4))
b = np.block([[A, B], [C, D]])
print(b)

# 按顺序垂直（逐行）堆叠数组。
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack((a,b))

# 按顺序水平（逐列）堆叠数组。
a = np.array([1, 2, 3,0]).reshape(2,2)
b = np.array([4, 5, 6,0]).reshape(2,2)
print(np.hstack((a, b)))
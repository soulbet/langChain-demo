import numpy as np
from numpy.random import default_rng

"""
NumPy 数组创建
"""
a1D = np.array([1, 2, 3, 4])
a2D = np.array([[1, 2], [3, 4]])
a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])


# 使用整数的开始、结束和步长值
rang_1 = np.arange(10)
rang_2 = np.arange(2, 10, dtype=float)
rang_3 = np.arange(2, 3, 0.1)

# 将创建具有指定元素数量的数组，并在指定的起始值和结束值之间均匀间隔
linspace_data = np.linspace(1., 4., 6)

#### 二维数组
a2D_0 = np.zeros((3, 3))
a2D_1 = np.ones((3, 3))
a2D_full = np.full((3, 3), 7)
# 定义了一个二维单位矩阵。其中 i=j（行索引和列索引相等）的元素为 1，其余元素为 0
eye_data = np.eye(3)

#### 随机数
# default_rng 结果的 random 方法将创建一个用 0 到 1 之间的随机值填充的数组
print(default_rng(42).random((2, 3)))

## numpy.indices 将创建一个数组集（堆叠成一个维度更高的数组），每个数组对应一个维度，并表示该维度上的变化
# 生成了网格点的坐标，比如 3 X 3的矩阵，会生成两个数组集，第一个表示行坐标，第二个表示列坐标
# 可以理解为，将坐标（x,y），拆分成了x和y两个数组
print(f"indices:{np.indices((3, 3))}")
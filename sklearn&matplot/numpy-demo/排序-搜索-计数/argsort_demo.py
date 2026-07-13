
import numpy as np

"""
沿给定轴执行间接排序，使用 kind 关键字指定的算法。它返回一个与 a 具有相同形状的索引数组，该数组沿给定轴按排序顺序索引数据。
"""
x = np.array([3, 1, 2])

print(np.argsort(x))

x = np.array([[0, 3], [2, 4]])
print(x)
# axis=1 按行排序   =0 按列排序
ind = np.argsort(x, axis=0)
print(ind)
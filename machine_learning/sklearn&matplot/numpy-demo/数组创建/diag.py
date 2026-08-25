

import numpy as np

"""
可以定义一个对角线上具有给定值的二维方阵，*或者* 如果给定一个二维数组，则返回一个仅包含对角线元素的以为数组

"""

# 提取对角线或构造对角线数组。
x = np.arange(9).reshape((3,3))
print(x)
print(np.diag(x))
print(np.diag(np.diag(x)))

# 构造对角线数组
print(np.diag([1,2,3,4]))

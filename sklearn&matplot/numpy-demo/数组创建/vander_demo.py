import numpy as np
"""
把数组里的每个元素，变成它的“各种次幂”，然后拼成一个矩阵
"""

x = np.array([1, 2, 3])
np.vander(x, 3)
print(np.linspace(0, 2, 5))

# N 输出矩阵的列数 如果 N 大于元素能达到的最高次幂，它会自动补 1；如果 N 小于最高次幂，它会截断高次幂
np.vander(np.linspace(0, 2, 5), 2)
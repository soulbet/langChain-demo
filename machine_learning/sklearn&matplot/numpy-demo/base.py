from sklearn import preprocessing

import numpy as np

x_train = np.array([[1., -1., 2.],
          [2., 0., 0.],
          [0., 1., -1.]])
print(f"维度数：{x_train.ndim}")
print(f"总元素数：{x_train.size}")
print(f"元素shape：{x_train.shape}")
print(f"区间内间隔：{np.linspace(0, 10, 5)}")
print(x_train[1][1])

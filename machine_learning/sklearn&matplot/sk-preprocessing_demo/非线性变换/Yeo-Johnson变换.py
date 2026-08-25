from sklearn.preprocessing import PowerTransformer
import numpy as np

"""
Yeo-Johnson变换
"""


X = np.array([[1, -2, 2],
              [-3, 1, 0],
              [0, 1, -1]])
# 假设 X 里面有正数、0、负数
pt = PowerTransformer(method='yeo-johnson', standardize=True)
# 注意这里的 standardize=True，它会自动在变换后帮你做一次 StandardScaler！一石二鸟！

X_transformed = pt.fit_transform(X)

# 你还可以看看它自动找到了什么幂次方
print("自动计算的最优lambda:", pt.lambdas_)

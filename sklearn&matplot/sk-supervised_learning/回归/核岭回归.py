from sklearn.kernel_ridge import KernelRidge
import numpy as np

"""
内核岭回归（KRR）将岭回归（具有L2-范数正则化的线性最小二乘）与内核技巧结合在一起。
因此，它学习了由各个内核和数据产生的空间中的线性函数。对于非线性内核，这对应于原始空间中的非线性函数。
"""

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = KernelRidge(alpha=1.0)
clf.fit(X, y)
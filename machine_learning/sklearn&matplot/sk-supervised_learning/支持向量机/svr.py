"""
回归
SVR 的思想是：允许预测值在真实值的 ±ε 范围内都不算错，只有超出这个管道的点才产生损失。
支持向量回归有三种不同的实现方式：SVR、NuSVR和LinearSVR。
LinearSVR提供了比SVR更快的实现，但只考虑了线性内核，而NuSVR实现的方式与SVR和LinearSVR略有不同
"""

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X, y)
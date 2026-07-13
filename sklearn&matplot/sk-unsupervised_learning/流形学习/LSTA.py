import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler

# 生成数据
X, color = make_swiss_roll(n_samples=800, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)

# LTSA 降维
ltsa = LocallyLinearEmbedding(n_components=2, n_neighbors=20, method='ltsa')
X_ltsa = ltsa.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(X_ltsa[:, 0], X_ltsa[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
plt.colorbar(label='原始位置（颜色）')
plt.title("LTSA 降维结果 (Swiss roll)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
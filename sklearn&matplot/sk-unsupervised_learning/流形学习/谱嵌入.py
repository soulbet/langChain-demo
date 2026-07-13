import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import SpectralEmbedding

# 1. 生成3D瑞士卷
X, color=make_swiss_roll(n_samples=1500, noise=0.0, random_state=42)

# 2. 谱嵌入（核心）
se=SpectralEmbedding(
    n_components=2,        # 降到2D
    affinity='nearest_neighbors',  # 用k-NN建图
    n_neighbors=15,        # 邻居数
    random_state=42
)
X_se=se.fit_transform(X)

# 3. 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_se[:, 0], X_se[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("谱嵌入（Spectral Embedding）结果")
plt.axis('equal')
plt.show()
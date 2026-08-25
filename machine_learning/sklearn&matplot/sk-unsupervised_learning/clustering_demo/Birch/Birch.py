from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成数据（10万点）
X, _ = make_blobs(n_samples=100000, centers=5, cluster_std=0.8, random_state=42)

#
# BIRCH 聚类
birch = Birch(
    threshold=0.5,        # 子簇直径阈值 threshold 控制叶子节点中每个 CF 子簇的最大直径
    branching_factor=50,  # 分支因子
    n_clusters=5          # 最终簇数（None 则输出子簇）
)
labels = birch.fit_predict(X)

# 结果
print(f"CF 树中的子簇数: {birch.subcluster_centers_.shape[0]}")
print(f"最终簇数: {len(set(labels))}")

# 可视化（采样显示）
idx = np.random.choice(X.shape[0], 5000, replace=False)
plt.figure(figsize=(10, 6))
plt.scatter(X[idx, 0], X[idx, 1], c=labels[idx], cmap='viridis', s=5, alpha=0.5)
plt.title(f'BIRCH 聚类结果 (n_samples=100k)')
plt.show()
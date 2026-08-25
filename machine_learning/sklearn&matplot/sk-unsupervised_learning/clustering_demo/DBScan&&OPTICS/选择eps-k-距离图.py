"""

对于每个点，计算到其第 k 个最近邻的距离（k = minPts），然后排序画图，寻找肘部拐点。
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import NearestNeighbors
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_blobs, _ = make_blobs(n_samples=200, centers=[[3, 3]], cluster_std=0.5, random_state=42)
X = np.vstack([X_moons, X_blobs])
minPts=2
k = minPts  # 通常 minPts = 2 * 维度
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# 第 k 个最近邻的距离（即到第 k 个邻居的距离）
k_distances = np.sort(distances[:, -1])

plt.plot(range(len(k_distances)), k_distances)
plt.xlabel('Points sorted by distance to k-th neighbor')
plt.ylabel(f'Distance to {k}-th nearest neighbor')
plt.title('k-distance Graph (Elbow for eps)')
plt.grid(True)
plt.show()
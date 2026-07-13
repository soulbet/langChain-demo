from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# 估计带宽（自动选择）
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=50)
print(f"估计的带宽: {bandwidth:.3f}")

# Mean Shift 聚类
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
y_pred = ms.fit_predict(X)

# 结果
print(f"簇数量: {len(ms.cluster_centers_)}")
print(f"簇中心:\n{ms.cluster_centers_}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30)
plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
            marker='x', c='red', s=200, linewidths=3, label='簇中心')
plt.legend()
plt.title('Mean Shift 聚类结果')
plt.show()
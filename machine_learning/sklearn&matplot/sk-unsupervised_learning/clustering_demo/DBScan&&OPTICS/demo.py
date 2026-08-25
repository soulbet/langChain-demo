from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成数据：半月形（非线性可分） + 噪声
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_blobs, _ = make_blobs(n_samples=200, centers=[[3, 3]], cluster_std=0.5, random_state=42)
X = np.vstack([X_moons, X_blobs])

# DBSCAN 聚类
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# 结果统计
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"估计的簇数: {n_clusters}")
print(f"噪声点数量: {n_noise}")

# 可视化
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = plt.cm.tab10.colors

for label in unique_labels:
    if label == -1:
        color = 'gray'
        marker = 'x'
        label_name = '噪声'
    else:
        color = colors[label % len(colors)]
        marker = 'o'
        label_name = f'簇 {label}'

    mask = (labels == label)
    plt.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                s=30 if label != -1 else 50, label=label_name, alpha=0.7)

plt.title('DBSCAN 聚类结果')
plt.legend()
plt.show()
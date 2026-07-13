from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# K-Means 聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# 结果
print(f"聚类中心:\n{kmeans.cluster_centers_}")
print(f"Inertia (簇内平方和): {kmeans.inertia_:.2f}")

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='x', c='red', s=200, linewidths=3)
plt.title("K-Means 聚类结果")
plt.show()
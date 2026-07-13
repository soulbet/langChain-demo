from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


# 衡量样本与自身簇的紧密度 vs 与其他簇的分离度，范围 [-1, 1]，越大越好

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K={k}, Silhouette Score={score:.3f}")
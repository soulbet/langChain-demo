from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

# 只计算最近邻的相似度

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

adj = kneighbors_graph(X, n_neighbors=30, mode='distance')
# 使用预计算的相似度矩阵
ap = AffinityPropagation(affinity='precomputed')
ap.fit(-adj.toarray())  # 距离转相似度（负值）
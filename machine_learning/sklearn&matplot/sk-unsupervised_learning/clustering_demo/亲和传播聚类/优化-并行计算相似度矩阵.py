from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
import joblib

# 使用 joblib 并行计算
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

with joblib.parallel_backend('loky', n_jobs=-1):
    S = -pairwise_distances(X, metric='sqeuclidean')
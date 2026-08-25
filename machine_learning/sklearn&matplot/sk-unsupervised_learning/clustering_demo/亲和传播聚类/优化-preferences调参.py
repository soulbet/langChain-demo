# 测试不同 preference 值
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
preferences = ['median', 'min', 'max', 0, 10, 100]
for pref in preferences:
    if isinstance(pref, str):
        ap = AffinityPropagation(preference=None, random_state=42)  # None 使用中位数
    else:
        ap = AffinityPropagation(preference=pref, random_state=42)
    ap.fit(X)
    print(f"preference={pref:>6}, 簇数={len(ap.cluster_centers_indices_)}")
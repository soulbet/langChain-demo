from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.utils import resample

# 随机采样减少数据量

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
X_sample = resample(X, n_samples=2000, random_state=42)
ap = AffinityPropagation(
    preference=None,           # 自动计算（中位数）
    damping=0.5,               # 阻尼系数
    max_iter=200,              # 最大迭代次数
    convergence_iter=15,       # 收敛所需的稳定迭代次数
    random_state=42
)
ap.fit(X_sample)
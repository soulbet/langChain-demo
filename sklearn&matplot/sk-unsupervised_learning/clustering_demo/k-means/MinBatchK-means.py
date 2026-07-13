from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
import time
import matplotlib.pyplot as plt

# 生成大数据集（10万样本）
X, _ = make_blobs(n_samples=100000, centers=5, random_state=42)

# 经典 K-Means（采样 10000 做对比）
from sklearn.cluster import KMeans

start = time.time()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)
time_kmeans = time.time() - start

# Mini-Batch K-Means
start = time.time()
mbk = MiniBatchKMeans(
    n_clusters=5,
    batch_size=1024,      # 小批量大小
    n_init=10,            # 初始化次数
    max_iter=100,         # 最大迭代次数
    random_state=42
)
mbk.fit(X)
time_mbk = time.time() - start

print(f"经典 K-Means 耗时: {time_kmeans:.2f} 秒")
print(f"Mini-Batch K-Means 耗时: {time_mbk:.2f} 秒")
print(f"加速比: {time_kmeans / time_mbk:.1f}x")

# 对比 inertia（近似程度）
print(f"经典 K-Means inertia: {kmeans.inertia_:.2f}")
print(f"Mini-Batch inertia: {mbk.inertia_:.2f}")
print(f"相对误差: {abs(kmeans.inertia_ - mbk.inertia_) / kmeans.inertia_ * 100:.2f}%")
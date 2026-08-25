from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# 亲和传播聚类
ap = AffinityPropagation(
    preference=None,           # 自动计算（中位数）
    damping=0.5,               # 阻尼系数
    max_iter=200,              # 最大迭代次数
    convergence_iter=15,       # 收敛所需的稳定迭代次数
    random_state=42
)

y_pred = ap.fit_predict(X)

# 查看结果
print(f"自动发现的簇数量: {ap.cluster_centers_indices_.shape[0]}")
print(f"簇中心索引: {ap.cluster_centers_indices_}")
print(f"迭代次数: {ap.n_iter_}")

# 可视化
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10.colors
for i, center_idx in enumerate(ap.cluster_centers_indices_):
    cluster_mask = (y_pred == i)
    plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                color=colors[i], alpha=0.6, label=f'簇 {i+1}')
    plt.scatter(X[center_idx, 0], X[center_idx, 1],
                color=colors[i], marker='*', s=200, edgecolors='black',
                label=f'中心 {i+1}')

plt.legend()
plt.title('Affinity Propagation 聚类结果')
plt.show()
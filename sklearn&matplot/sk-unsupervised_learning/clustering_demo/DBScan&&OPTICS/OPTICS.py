from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成密度不均匀的数据
X1, _ = make_blobs(n_samples=200, centers=[[0,0]], cluster_std=0.2, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[3,3]], cluster_std=0.5, random_state=42)
X3, _ = make_blobs(n_samples=50, centers=[[-2,3]], cluster_std=0.1, random_state=42)
X = np.vstack([X1, X2, X3])
"""
min_samples	核心点最小邻居数（同 DBSCAN 的 minPts）	5-20	关键参数
max_eps	最大半径（ε_max）	∞ 或数据尺度	性能优化，设太大则慢
xi	提取聚类的陡峭度阈值	0.01-0.1	自动提取，越小簇越多
min_cluster_size	最小簇大小	0.05（比例）	过滤小簇
metric	距离度量	'euclidean'	可选 'manhattan', 'cosine'

"""
# OPTICS 聚类
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
labels = optics.fit_predict(X)

# 结果
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"估计的簇数: {n_clusters}")
print(f"噪声点数量: {n_noise}")

# 可视化
plt.figure(figsize=(14, 5))

# 子图1：聚类结果
plt.subplot(1, 2, 1)
unique_labels = set(labels)
colors = plt.cm.tab10.colors
for label in unique_labels:
    if label == -1:
        color = 'gray'
        marker = 'x'
    else:
        color = colors[label % len(colors)]
        marker = 'o'
    mask = (labels == label)
    plt.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=20, alpha=0.7)
plt.title('OPTICS 聚类结果')
plt.xlabel('X1')
plt.ylabel('X2')

# 子图2：可达性图
plt.subplot(1, 2, 2)
reachability = optics.reachability_[optics.ordering_]
plt.plot(range(len(reachability)), reachability, 'b-')
plt.xlabel('点的处理顺序')
plt.ylabel('可达距离')
plt.title('OPTICS 可达性图')
plt.axhline(y=optics.xi_, color='r', linestyle='--', label=f'xi阈值={optics.xi_}')
plt.legend()

plt.tight_layout()
plt.show()

# 使用不同的 eps 阈值提取聚类
for eps_cut in [0.3, 0.5, 0.8, 1.2]:
    # 基于可达性图手动提取（近似）
    # 实际使用 optics 对象的 cluster_hierarchy_ 属性
    print(f"eps_cut={eps_cut}: 需要重新运行不同参数")
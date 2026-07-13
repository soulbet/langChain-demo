"""
谱双聚类（Spectral Co-clustering）示例程序

本程序演示如何使用谱双聚类算法对合成数据进行聚类分析，
包括数据生成、打乱、聚类和结果可视化。
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

# 生成合成双聚类数据集，包含5个簇，添加噪声
data, rows, columns = make_biclusters(
    shape=(300, 300), n_clusters=5, noise=5,
    shuffle=False, random_state=0)

# 可视化原始数据集
# 在新图形窗口中将二维数组显示为矩阵
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# 随机打乱数据的行和列，模拟真实场景中的无序数据
# 局部独立随机生成器,随机数种子 0 ，可以换成其他的，只是随机数队列不同
rng = np.random.RandomState(0)
# 随机排列一个序列，或返回一个排列过的范围。
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

# 可视化打乱后的数据集
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

# 创建谱双聚类模型并拟合数据
model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)

# 计算共识分数，评估聚类结果与真实标签的一致性
score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.3f}".format(score))

# 根据聚类结果对数据进行重排，使同一簇的数据相邻
fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

# 可视化双聚类结果
plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()

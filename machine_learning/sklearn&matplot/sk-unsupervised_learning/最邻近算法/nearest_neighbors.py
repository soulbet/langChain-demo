
"""
它是一个无监督的模型，不关心样本的标签，唯一的目标就是找到离得最近的邻居

"""
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 假设这是我们数据库中的商品向量 (5个商品，每个由3个特征表示)
X_db = np.array([[0, 0, 2], [1, 0, 0], [0, 0, 1], [1, 1, 1], [2, 0, 0]])

# 初始化模型 (只找最近的2个邻居)
nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean')
nbrs.fit(X_db)

# 来了一个查询的商品向量
X_query = [[0, 0, 1.3]]

# 找到它的2个最近邻
distances, indices = nbrs.kneighbors(X_query)

print("最相似商品的索引:", indices)   # 输出例如 [[2 0]]
print("对应的距离:", distances)      # 输出例如 [[0.3 0.7]]
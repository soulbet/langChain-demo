from sklearn.neighbors import NearestNeighbors
import numpy as np

"""


KD树 (K-Dimensional Tree)：用垂直于坐标轴的超平面（可以理解为高维空间中的“墙壁”）来切分空间，将数据点递归地放入一个个不重叠的矩形框（或超矩形）中。
在低维数据（通常认为维度 < 20）上，效率极高
Ball树 (Ball Tree)：用一系列嵌套的超球体来包裹数据点。每个节点定义一个球心和一个半径，将数据点划分到不同的球体中，且球体之间可以重叠。
对中高维数据（20 - 50维）表现更稳定，在高维空间下通常比KD树更快

kd-tree：适合低维数据
ball-tree：适合高维数据
暴力搜索：小数据集
"""

# 1. 创建数据
X = np.random.random((10000, 10))

# 2. 关键步骤：设置 algorithm='auto'
# 让 sklearn 自己决定是用 KDTree、BallTree 还是暴力搜索
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto')
nbrs.fit(X)

# 3. 查询
distances, indices = nbrs.kneighbors([X[0]])
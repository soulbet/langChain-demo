import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

# 生成弯曲数据（瑞士卷）
X, color = make_swiss_roll(n_samples=1500, noise=0.0, random_state=42)

# ====================== MLLE 核心代码 ======================
mlle = LocallyLinearEmbedding(
    n_neighbors=50,        # 邻居数（比HE小很多）
    n_components=2,        # 降到2维
    method='modified',    # ✅ 这个就是 MLLE！
    random_state=42
)

# 降维
X_mlle = mlle.fit_transform(X)

# 画图
plt.figure(figsize=(8,6))
plt.scatter(X_mlle[:,0], X_mlle[:,1], c=color, cmap=plt.cm.Spectral)
plt.title("MLLE 降维结果")
plt.axis('equal')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

# ====================== 1. 生成经典弯曲数据（瑞士卷）======================
X, color = make_swiss_roll(n_samples=1500, noise=0.0, random_state=42)

# ====================== 2. HE = Hessian LLE 降维 ======================
# method='hessian' 就是 HE！
he = LocallyLinearEmbedding(
    n_neighbors=120,       # HE 对 k 要求高，必须比 LLE 大
    n_components=2,        # 降到 2 维
    method='hessian',      # ✅ 这个就是 HE（Hessian LLE）
    eigen_solver='auto'
)

X_he = he.fit_transform(X)

# ====================== 3. 画图对比 ======================
plt.figure(figsize=(8, 6))
plt.scatter(X_he[:, 0], X_he[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title('HE (Hessian LLE) 降维结果')
plt.axis('equal')
plt.show()
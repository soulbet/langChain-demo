from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import make_swiss_roll

# 生成数据（如 Swiss roll）
X, _ = make_swiss_roll(n_samples=1000, noise=0.1)

# 应用LLE（降至2维）
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)

# 可视化结果
import matplotlib.pyplot as plt
plt.scatter(X_lle[:, 0], X_lle[:, 1], c='b', s=10)
plt.title("LLE on Swiss Roll")
plt.show()

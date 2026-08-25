import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 加载数据（手写数字 8×8=64 维）
digits = load_digits()
X, y = digits.data, digits.target
print(f"原始形状: {X.shape}")  # (1797, 64)

# 标准化
X_scaled = StandardScaler().fit_transform(X)

# PCA 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='数字类别')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('手写数字 PCA 降维可视化')
plt.show()

# 选择主成分数（保留 95% 方差）
pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X_scaled)
print(f"保留95%方差后的维度: {X_95.shape[1]}")  # 通常 64→20~30
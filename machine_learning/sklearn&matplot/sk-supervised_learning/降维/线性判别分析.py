
"""
目标：在有监督的情况下，找到一个投影方向，使得投影后，同类样本尽可能接近，不同类样本尽可能远离。它的核心是最大化类间散度与类内散度的比值。

核心思想：寻找一个方向，让不同类别之间的区分度最大。
有监督
类间方差  越大越好，说明分得开
类内方差  越小越好，说明同类很紧凑

最佳投影方向 = 类间方差 / 类内方差
1、只能降k-1维
2、监督算法，必须有标签
3、样本数大于特征数
4、也能作为分类器

"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载经典鸢尾花数据（3个类别，4个特征）
iris = load_iris()
X = iris.data
y = iris.target

# 黄金法则：LDA 对量纲极其敏感，必须缩放！
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------- 使用 PCA 降维到 2D --------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -------- 使用 LDA 降维到 2D --------
# 注意：LDA 的 fit 必须同时传入 X 和 y！
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# -------- 画图对比 --------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PCA 图
for i, color in zip([0,1,2], ['r', 'g', 'b']):
    axes[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=color, label=iris.target_names[i])
axes[0].set_title("PCA 降维 (无监督)")
axes[0].legend()

# LDA 图
for i, color in zip([0,1,2], ['r', 'g', 'b']):
    axes[1].scatter(X_lda[y == i, 0], X_lda[y == i, 1], c=color, label=iris.target_names[i])
axes[1].set_title("LDA 降维 (有监督)")
axes[1].legend()

plt.show()

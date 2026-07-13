
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

# 假设有 6 个人的年龄（注意长尾：最后一个人是 100 岁）
X = np.array([[20], [22], [25], [28], [30], [100]])

# ---------------------------
# 1. 等宽分箱 - 会受异常值影响
# ---------------------------
"""
原理：把数据的取值范围等分。比如年龄从 0 到 100，切 5 个箱，每个箱宽度是 20（0-20, 20-40…）。
致命缺点：极其怕长尾和离群值！ 如果有一个 1000 岁的人，你的前几个箱全被挤在极小的范围内，毫无意义。
适用场景：数据分布非常均匀，且没有异常值时。
"""
est_wide = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_wide = est_wide.fit_transform(X)
print("等宽分箱边界:", est_wide.bin_edges_)
# 边界大约是: [20, 46.6, 73.3, 100]。前4个人全被挤在第一个箱里了！
print("等宽分箱结果:\n", X_wide.flatten())
# 输出: [0. 0. 0. 0. 1. 2.]

# ---------------------------
# 2. 等频分箱 - 强烈推荐，不受异常值影响
# ---------------------------
"""
原理：保证每个箱里的样本数量大致一样多。也就是按分位数切（比如前 20% 的人一个箱，20%-40% 一个箱）。
优点：完美避开了长尾和异常值的干扰。即使最大值是 1000，它也仅仅是孤零零地落在最后一个箱里，不会影响前面箱的边界。
适用场景：绝大多数工业界场景，特别是带有长尾分布的数据（如收入、交易额）。
"""
est_freq = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_freq = est_freq.fit_transform(X)
print("等频分箱边界:", est_freq.bin_edges_)
# 边界大约是: [20, 23.6, 29, 100]。完美保护了前面的正常数据！
print("等频分箱结果:\n", X_freq.flatten())
# 输出: [0. 0. 1. 1. 2. 2.] 每个箱里基本都是2个人

# ---------------------------
# 3. 聚类分箱
# ---------------------------
"""
原理：用 K-Means 聚类算法，在数据密集的地方切得密一点，在数据稀疏的地方切得疏一点。
优点：非常贴合数据本身的空间分布。
缺点：计算量大，速度慢，且每次跑可能切的边界不一样（有随机性）。
"""
est_kmeans = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
X_kmeans = est_kmeans.fit_transform(X)
print("聚类分箱结果:\n", X_kmeans.flatten())

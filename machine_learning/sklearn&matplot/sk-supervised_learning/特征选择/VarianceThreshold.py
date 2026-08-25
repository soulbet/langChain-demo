"""

一个最简单的特征选择工具：直接删掉 “几乎不变化、没信息量” 的特征。
计算每个特征的方差：
方差大 → 数值变化多 → 有用
方差小 → 数值几乎一样 → 没用
"""
from sklearn.feature_selection import VarianceThreshold

# 数据（每行一个样本，每列一个特征）
X = [
    [0, 1, 0],
    [0, 2, 0],
    [0, 3, 0],
    [0, 4, 0]
]

# 阈值：方差>0.01才保留
sel = VarianceThreshold(threshold=0.01)

# 过滤特征
X_new = sel.fit_transform(X)

print("过滤前形状：", X.shape)
print("过滤后形状：", X_new.shape)
print("保留的特征：\n", X_new)
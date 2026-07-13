from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

"""
先训练一个模型（随机森林 / 逻辑回归 / 线性 SVM / GBDT）
模型会自动算出每个特征的重要性
SelectFromModel 自动保留重要性高于阈值的特征
输出筛选后的 X
与RFE比，只训练一次，速度快
"""

X, y = load_iris(return_X_y=True)
# 1. 训练一个模型
model = RandomForestClassifier()
model.fit(X, y)

# 2. 用模型自动筛选特征
# threshold="mean"：保留 大于平均值 的重要特征
# threshold="median"：保留 大于中位数 的重要特征
# threshold=0.05：保留重要性 >0.05 的特征
sfm = SelectFromModel(model)
X_selected = sfm.fit_transform(X, y)

print("原始特征数：", X.shape[1])
print("筛选后特征数：", X_selected.shape[1])
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
X, y = load_iris(return_X_y=True)
# 基模型（必须能输出特征重要性/系数）
model = RandomForestClassifier()
# 获取特征重要度
print(model.feature_importances_)
# RFE：最后保留 5 个特征
rfe = RFE(estimator=model, n_features_to_select=5)

X_selected = rfe.fit_transform(X, y)  # 训练+筛选
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

"""
协同训练（简单版 Co-Training）
两个模型用不同特征，互相补充标签
"""
X, y = load_iris(return_X_y=True)
# 构造半监督：部分标签置-1代表无标注
y_semi = y.copy()
y_semi[30:] = -1
# 特征拆分两组（模拟Co-Training）
X1 = X[:, :2]
X2 = X[:, 2:]

# 模型A、模型B
clf1 = LogisticRegression()
clf2 = LogisticRegression()

# 第一轮用有标签训练
labeled = y_semi != -1
clf1.fit(X1[labeled], y[labeled])
clf2.fit(X2[labeled], y[labeled])

# 互相给高置信度样本打标签
y1 = clf1.predict_proba(X1).max(axis=1)
y2 = clf2.predict_proba(X2).max(axis=1)

print("协同训练完成（双模型互补）")
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

"""
SelfTraining（自训练）
用有标签训模型 → 预测无标签 → 置信度高的加入训练 → 循环
"""
# 数据
X, y = load_iris(return_X_y=True)
y_semi = y.copy()
y_semi[20:] = -1  # 后面全部无标签

# 基模型 + 自训练
base_clf = RandomForestClassifier()
self_clf = SelfTrainingClassifier(base_clf)
self_clf.fit(X, y_semi)

# 结果
print("自训练预测标签：")
print(self_clf.transduction_)
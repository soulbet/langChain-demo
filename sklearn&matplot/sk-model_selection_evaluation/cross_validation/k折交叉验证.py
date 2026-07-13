from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 1. 定义模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. 执行5折交叉验证
# scoring='accuracy' 指定评估指标为准确率
# cv=5 指定5折
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"各折准确率: {scores}")
print(f"平均准确率: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
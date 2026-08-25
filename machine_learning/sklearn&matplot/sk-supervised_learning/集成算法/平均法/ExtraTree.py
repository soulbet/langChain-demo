from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# 极端随机树
et = ExtraTreesClassifier(
    n_estimators=100,  # 树数量
    max_depth=None,
    random_state=42
)
et.fit(X, y)
print(et.score(X, y))
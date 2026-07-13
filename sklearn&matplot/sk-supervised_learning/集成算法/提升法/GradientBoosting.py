from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
model = GradientBoostingClassifier(
    n_estimators=100,    # 树的数量（迭代次数）
    learning_rate=0.1,   # 学习率：每棵树贡献多少
    max_depth=3,         # 每棵树深度，防止过拟合
    subsample=0.8,       # 随机采样样本（随机梯度提升）
    random_state=42
)
model.fit(X_train, y_train)
GradientBoostingClassifier(random_state=0)
model.predict(X_test[:2])

model.score(X_test, y_test)
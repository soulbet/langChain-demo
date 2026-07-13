from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
clf = HistGradientBoostingClassifier(
    max_iter=100,       # 迭代次数（树数量）
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
clf.fit(X_train, y_train)

# 类别特征处理，会将相同/相似的特征放一个组
# model.fit(X, y, categorical_features=cat_features)
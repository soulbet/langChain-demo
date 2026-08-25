from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
# 底层基模型
base_estimators = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]
# 元模型
meta_clf = LogisticRegression()

stack_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_clf,
    cv=5  # 5折交叉验证生成中间特征
)
stack_clf.fit(X_train, y_train)
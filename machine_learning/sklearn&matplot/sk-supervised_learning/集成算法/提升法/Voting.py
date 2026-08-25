from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
# 定义多个不同模型
model1 = LogisticRegression()
model2 = DecisionTreeClassifier(max_depth=3)
model3 = SVC(probability=True)

# 投票集成
voting = VotingClassifier(
    estimators=[('lr', model1), ('tree', model2), ('svm', model3)],
    voting='soft',  # 软投票（用概率）
    weights=[1,1,1] # 每个模型的权重
)

voting.fit(X_train, y_train)
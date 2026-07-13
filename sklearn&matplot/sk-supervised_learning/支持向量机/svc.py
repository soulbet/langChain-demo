from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 生成简单数据
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练 SVM
clf = svm.SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(X_train, y_train)

# 预测与评估
print("Accuracy:", clf.score(X_test, y_test))
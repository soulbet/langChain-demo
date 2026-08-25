from sklearn.neighbors import  KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据准备
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征缩放（必须！KNN 对尺度敏感）
scaler = StandardScaler()
# 学习训练数据的均值/标准差，并同时将训练数据标准化
X_train = scaler.fit_transform(X_train)
# 用训练数据学到的参数，只转换测试数据（不再重新学习）
X_test = scaler.transform(X_test)

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='uniform')
knn_reg.fit(X_train, y_train)
y_pred_reg = knn_reg.predict(X_test)
print(f"R²: {knn_reg.score(X_test, y_test):.3f}")